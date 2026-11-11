import os, sys, matplotlib, logging
matplotlib.use("Agg"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
BG, Y, G, L = "#000000", "#FFD60A", "#8A8A8A", "#EDEDED"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG, "text.color": L, "mathtext.fontset": "dejavusans",
                     "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"]})


def page(name, rows, h):
    fig = plt.figure(figsize=(13, h)); y = 1.0; step = 1.0 / len(rows)
    for label, eq in rows:
        fig.text(0.01, y - step * 0.5, label, color=Y, fontsize=15, fontweight="bold", va="center")
        fig.text(0.27, y - step * 0.5, eq, color=L, fontsize=17, va="center")
        y -= step
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


page("m1_model.png", [
    ("Pose (6 numbers)", r"$p=(\mathbf{r},\mathbf{t}),\quad R=\exp([\mathbf{r}]_\times)\ \mathrm{(Rodrigues)},\quad T(p)=\left[\begin{smallmatrix}R & \mathbf{t}\\ 0 & 1\end{smallmatrix}\right]$".replace(r"\left[\begin{smallmatrix}R & \mathbf{t}\\ 0 & 1\end{smallmatrix}\right]", r"[R\ |\ \mathbf{t}]")),
    ("Projection", r"$\tilde{u}_{ki}=P_k\,T(p)\,\tilde{X}_i,\qquad u_{ki}(p)=(\tilde{u}_1/\tilde{u}_3,\ \tilde{u}_2/\tilde{u}_3)$" + "\n" + r"$P_k\in\mathbb{R}^{3\times4}$ = view $k$,   $X_i$ = CAD fiducial $i=1..5$ (mm)"),
    ("Simulated detections", r"$\mathcal{Y}_k=\mathrm{shuffle}\left(\{u_{ki}(p^*)+\varepsilon_{ki}:\ i\in V_k\}\ \cup\ C_k\right),\quad \varepsilon\sim\mathcal{N}(0,\sigma^2 I)$" + "\n" + r"$V_k$ = visible fiducials (missing = removed),   $C_k$ = clutter, uniform near the cluster"),
    ("True pose", r"$p^*$: rotation $20^\circ$ about a random axis, translation 15 mm in a random direction"),
    ("Stage 1", r"$\mathbf{r}_0=\mathbf{r}^*+\delta,\ \delta\sim U[-\Delta,\Delta]^3;\quad \mathbf{t}_0=\arg\min_{\mathbf{t}}\ \sum_k \Vert\, \pi_k(R_0\bar{X}+\mathbf{t})-\bar{y}_k \Vert^2$"),
    ("Stage 2 (CMA-ES)", r"$p=p_0+S\,z,\ \ S=\mathrm{diag}(15^\circ,15^\circ,15^\circ,5,5,5\,\mathrm{mm}),\ \ z\sim\mathcal{N}(m,\sigma^2C),\ \ \lambda=100$"),
], 6.2)

page("m2_costs.png", [
    ("Distances", r"$d_{kij}(p)=\Vert u_{ki}(p)-y_{kj}\Vert,\quad y_{kj}\in\mathcal{Y}_k,\ \ M_k=|\mathcal{Y}_k|$"),
    ("Repo cost\n(loops over 5 fiducials)", r"$f_{\mathrm{repo}}(p)=\sqrt{\frac{1}{2NK}\sum_k\sum_{i=1}^{N}\ \min_j\ d_{kij}^2}$"),
    ("Detections → fiducials", r"$f_{\mathrm{rev}}(p)=\sqrt{\frac{1}{2\sum_k M_k}\sum_k\sum_{j=1}^{M_k}\ \min_i\ d_{kij}^2}$"),
    ("Best m per view\n(minimum fiducials)", r"$f_{m}(p)=\sqrt{\frac{1}{2\sum_k m_k}\sum_k\sum_{l=1}^{m_k} e_{k(l)}^2},\quad e_{ki}=\min_j d_{kij},\ \ m_k=\min(m,M_k)$" + "\n" + r"$e_{k(1)}\leq e_{k(2)}\leq\ldots$ sorted; $m=5$ gives the repo cost"),
    ("Why the repo cost fails", r"if fiducial $i$ is not detected in view $k$:  $e_{ki}(p^*)\approx$ spacing between balls (90–135 px)" + "\n" + r"squared loss $\Rightarrow$ cheaper to shift every ball a little than leave one 100 px residual"),
], 6.0)

page("m3_metrics.png", [
    ("TRE at a point", r"$\mathrm{TRE}(x)=\Vert\,\hat{T}x-T^*x\,\Vert\quad$ (mm)"),
    ("RMS TRE (per trial)", r"$\mathrm{TRE}_{\mathrm{RMS}}=\sqrt{\frac{1}{|G|}\sum_{x\in G}\mathrm{TRE}(x)^2},\ \ G$ = 10×10×10 grid, ~300 mm cube" + "\n" + r"success: $\mathrm{TRE}_{\mathrm{RMS}}<2$ mm"),
    ("Rotation / translation", r"$\theta=\arccos\frac{\mathrm{tr}(\hat{R}R^{*\top})-1}{2},\qquad e_t=\mathrm{TRE}(\bar{X})$  (at fiducial centroid)"),
    ("View-pair consistency\n(no ground truth)", r"$s=\max_{a<b}\ \sqrt{\frac{1}{|G|}\sum_{x\in G}\Vert\hat{T}_{a}x-\hat{T}_{b}x\Vert^2}$,   $\hat{T}_a$ = solve using only view pair $a$"),
    ("Detector image model", r"$A(u)=A_{\mathrm{body}}+A_{\mathrm{bone}}+\sum_i\mu_g\,\ell_i(u),\ \ \ell_i$ = chord through ball $i$" + "\n" + r"$I\sim\mathrm{Poisson}(I_0e^{-A}),\quad \mathrm{image}=-\ln(I/I_0)$"),
    ("Detector", r"LoG at $s=r/\sqrt{2}$ $\to$ candidates $\to$ fit $a\sqrt{\rho^2-\Vert u-c\Vert^2}+$quadratic background" + "\n" + r"keep if $|\rho/r-1|<0.25$ and $a\rho\in[0.5,1.8]\,\mu_g\cdot10$ mm;  output $c$"),
], 6.4)
print("math figures written")
