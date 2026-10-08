"""Benchmark: gather/index_add recurrent product (current nfly) vs cuSPARSE CSR SpMM + SDDMM. Checks gradients match."""
import time, sys, torch
sys.path.insert(0, "/nesi/project/aut04653/Manj/Fly/nfly")
from nfly.brain.rnn import _SparseRecurrent
torch.manual_seed(0); dev = "cuda"
N, E, B, T = 34000, 2_870_000, int(sys.argv[1]) if len(sys.argv) > 1 else 64, 344
pre = torch.randint(N, (E,), device=dev); post = torch.randint(N, (E,), device=dev)

class CSRRec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, S):
        Wf = torch.sparse_csr_tensor(S["crow_f"], S["col_f"], w[S["perm_f"]], (N, N))
        ctx.save_for_backward(h, w); ctx.S = S
        return torch.sparse.mm(Wf, h.t()).t()
    @staticmethod
    def backward(ctx, g):
        h, w = ctx.saved_tensors; S = ctx.S; gh = gw = None
        if ctx.needs_input_grad[0]:
            Wb = torch.sparse_csr_tensor(S["crow_b"], S["col_b"], w[S["perm_b"]], (N, N))
            gh = torch.sparse.mm(Wb, g.t()).t()
        if ctx.needs_input_grad[1]:
            pat = torch.sparse_csr_tensor(S["crow_f"], S["col_f"], torch.zeros(E, device=g.device), (N, N))
            v = torch.sparse.sampled_addmm(pat, g.t().contiguous(), h.contiguous(), beta=0.0).values()
            gw = torch.empty_like(w); gw[S["perm_f"]] = v
        return gh, gw, None

def build(pre, post):
    S = {}
    for tag, r, c in (("f", post, pre), ("b", pre, post)):
        key = r.long() * N + c.long(); perm = torch.argsort(key)
        S["perm_" + tag] = perm; S["col_" + tag] = c[perm].int()
        S["crow_" + tag] = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), torch.bincount(r, minlength=N).cumsum(0)]).int()
    return S
S = build(pre, post)

def run(kind, w, h0):
    h = h0
    for t in range(T):
        x = (_SparseRecurrent.apply(h, w, pre, post, 2_000_000) if kind == "old" else CSRRec.apply(h, w, S))
        h = 0.9 * h + 0.1 * torch.relu(x + 0.02).clamp(max=10)
    return h.pow(2).mean()

w0 = (torch.rand(E, device=dev) * 0.02) * torch.where(torch.rand(E, device=dev) < 0.7, 1.0, -1.0)
h0 = torch.rand(B, N, device=dev)
res = {}
for kind in ("old", "csr", "old", "csr"):
    w = w0.clone().requires_grad_(True); torch.cuda.synchronize(); t0 = time.time()
    L = run(kind, w, h0); L.backward(); torch.cuda.synchronize()
    res[kind] = (L.item(), w.grad.clone(), time.time() - t0)
    print(f"{kind}: loss {L.item():.6f}  fwd+bwd {time.time()-t0:.2f}s  ({T} steps, B={B})", flush=True)
g1, g2 = res["old"][1], res["csr"][1]
print("grad rel err", ((g1 - g2).norm() / g1.norm()).item(), " speedup", res["old"][2] / res["csr"][2])
