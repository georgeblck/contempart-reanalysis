"""Generate README result tables from the CSVs in `results/`.

Reads:
    results/heads.csv
    results/all_mantel_permanova.csv
    results/all_dbrda.csv
    results/all_varpart.csv
    results/all_social.csv

Writes:
    results/readme_tables.md    markdown tables for copy-paste or include
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path("results")


PAPER_LINKS: dict[str, str] = {
    "CLIP ViT-L/14 (OpenAI)": "[Radford 2021](https://arxiv.org/abs/2103.00020)",
    "CLIP ViT-B/32 (OpenAI)": "[Radford 2021](https://arxiv.org/abs/2103.00020)",
    "OpenCLIP ViT-L/14 (LAION-2B)": "[Cherti 2023](https://arxiv.org/abs/2212.07143)",
    "DINOv2 ViT-L/14": "[Oquab 2024](https://arxiv.org/abs/2304.07193)",
    "CSD ViT-L/14": "[Somepalli 2024](https://arxiv.org/abs/2404.01292)",
    "SCFlow (ICCV 2025)": "[Ma 2025](https://arxiv.org/abs/2503.11478)",
    "GOYA MLP (on CLIP-B32)": "[Gou 2023](https://arxiv.org/abs/2305.13770)",
    "OpenCLIP ViT-H/14 (LAION-2B)": "[Kim 2025](https://arxiv.org/abs/2503.13531)",
    "SD 2.1 VAE": "[Rombach 2022](https://arxiv.org/abs/2112.10752)",
    "SSCD ResNet50": "[Pizzi 2022](https://arxiv.org/abs/2202.10261)",
}

# Per-head overrides (used when the recipe itself has a separate citation).
PAPER_LINKS_BY_HEAD: dict[str, str] = {
    "dinov2_style": "[Oquab 2024](https://arxiv.org/abs/2304.07193) + [fruit-SALAD (Schaldenbrand 2024)](https://arxiv.org/abs/2406.01278)",
}

CODE_LINKS: dict[str, str] = {
    "CLIP ViT-L/14 (OpenAI)": "[openai/CLIP](https://github.com/openai/CLIP)",
    "CLIP ViT-B/32 (OpenAI)": "[openai/CLIP](https://github.com/openai/CLIP)",
    "OpenCLIP ViT-L/14 (LAION-2B)": "[mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)",
    "DINOv2 ViT-L/14": "[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)",
    "CSD ViT-L/14": "[learn2phoenix/CSD](https://github.com/learn2phoenix/CSD)",
    "SCFlow (ICCV 2025)": "[compvis/scflow](https://github.com/CompVis/scflow)",
    "GOYA MLP (on CLIP-B32)": "[yankungou/GOYA](https://github.com/yankungou/GOYA)",
    "OpenCLIP ViT-H/14 (LAION-2B)": "[mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)",
    "SD 2.1 VAE": "[huggingface.co/sd2-community](https://huggingface.co/sd2-community/stable-diffusion-2-1)",
    "SSCD ResNet50": "[facebookresearch/sscd](https://github.com/facebookresearch/sscd-copy-detection)",
}


ROLE_ORDER = ["content", "style", "general", "appearance", "duplicate"]


def sig(p: float) -> str:
    return "✓" if p < 0.05 else " "


def ordered_heads() -> pd.DataFrame:
    """Registry sorted by role group, registry order preserved within each group."""
    heads = pd.read_csv(RESULTS / "heads.csv")
    heads["_role_rank"] = heads["role"].map({r: i for i, r in enumerate(ROLE_ORDER)})
    heads["_idx"] = heads.index
    return heads.sort_values(["_role_rank", "_idx"]).drop(columns=["_role_rank", "_idx"]).reset_index(drop=True)


def embedding_catalog() -> str:
    heads = ordered_heads()
    lines = [
        "| Head | Role | Backbone | Dim | Paper | Code |",
        "|---|---|---|---:|---|---|",
    ]
    for _, r in heads.iterrows():
        paper = PAPER_LINKS_BY_HEAD.get(r["name"]) or PAPER_LINKS.get(r.backbone, "-")
        code = CODE_LINKS.get(r.backbone, "-")
        lines.append(
            f"| **{r.display}** | {r.role} | {r.backbone} | {r.dim} | {paper} | {code} |"
        )
    return "\n".join(lines)


def dbrda_table() -> str:
    """db-RDA: variance % controlling for all other variables. Cols sorted by mean effect."""
    path = RESULTS / "all_dbrda.csv"
    if not path.exists():
        return "_(run `Rscript R/dbrda.R` to populate this table)_"
    df = pd.read_csv(path)
    heads = ordered_heads()

    variables = ["professor_class", "continent", "gender", "school"]
    var_display = {"professor_class": "professor", "continent": "continent",
                   "gender": "gender", "school": "school"}

    # Per-column max value for tie-aware bolding. Independent of significance:
    # the checkmark already carries p < 0.05; bolding highlights effect size.
    max_pct_per_var: dict[str, float | None] = {}
    for v in variables:
        rows = df[df["variable"] == v]
        max_pct_per_var[v] = float(rows["variance_pct"].max()) if not rows.empty else None

    lines = [
        "| Head | Role |" + "".join(f" {var_display[v]} |" for v in variables),
        "|---|---|" + "---:|" * len(variables),
    ]
    for _, h in heads.iterrows():
        sub = df[df["head"] == h["name"]]
        cells = [f"**{h.display}**", h.role]
        for v in variables:
            row = sub[sub["variable"] == v]
            if row.empty:
                cells.append("-")
                continue
            pct = row["variance_pct"].iloc[0]
            p = row["p_value"].iloc[0]
            val = f"{pct:.1f}% {sig(p)}"
            max_pct = max_pct_per_var[v]
            if max_pct is not None and round(pct, 1) == round(max_pct, 1):
                val = f"**{val.strip()}**"
            cells.append(val)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def varpart_table() -> str:
    path = RESULTS / "all_varpart.csv"
    if not path.exists():
        return "_(run `Rscript R/dbrda.R` to populate this table)_"
    df = pd.read_csv(path)
    heads = ordered_heads()
    lines = [
        "| Head | Professor only | Shared (school ∩ prof) | School only | Residual |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, h in heads.iterrows():
        sub = df[df["head"] == h["name"]]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"| **{h.display}** | {r.professor_only * 100:.1f}% | {r.shared * 100:.1f}% | "
            f"{r.school_only * 100:.1f}% | {r.residual * 100:.1f}% |"
        )
    return "\n".join(lines)


def social_table() -> str:
    df = pd.read_csv(RESULTS / "all_social.csv")
    heads = ordered_heads()
    graphs = ["GU", "GY"]
    graph_display = {"GU": "G^U (artist-to-artist)", "GY": "G^Y (full network)"}

    # Per-column max r (among heads, excluding VGG-2020 row). Independent of
    # significance: ✓ already carries p < 0.05, bolding highlights effect size.
    head_names = set(heads["name"])
    max_r_per_graph: dict[str, float | None] = {}
    for g in graphs:
        rows = df[(df["graph"] == g) & (df["head"].isin(head_names))]
        max_r_per_graph[g] = float(rows["mantel_r"].max()) if not rows.empty else None

    def render_cell(r: float, p: float, max_r: float | None) -> str:
        s = f"r = {r:+.3f} {sig(p)}".rstrip()
        if max_r is not None and round(r, 3) == round(max_r, 3):
            return f"**{s}**"
        return s

    lines = [
        "| Head | Role |" + "".join(f" {graph_display[g]} |" for g in graphs),
        "|---|---|" + "---|" * len(graphs),
    ]
    for _, h in heads.iterrows():
        sub = df[df["head"] == h["name"]]
        if sub.empty:
            continue
        cells = [f"**{h.display}**", h.role]
        for g in graphs:
            row = sub[sub["graph"] == g]
            if row.empty:
                cells.append("-")
                continue
            r = row["mantel_r"].iloc[0]
            p = row["mantel_p"].iloc[0]
            cells.append(render_cell(r, p, max_r_per_graph[g]))
        lines.append("| " + " | ".join(cells) + " |")
    vgg = df[df["head"] == "_vgg_2020"]
    if not vgg.empty:
        cells = ["**VGG-2020** *(2020 paper baseline)*", "—"]
        for g in graphs:
            row = vgg[vgg["graph"] == g]
            if row.empty:
                cells.append("-")
                continue
            r = row["mantel_r"].iloc[0]
            p = row["mantel_p"].iloc[0]
            cells.append(render_cell(r, p, None))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    out = RESULTS / "readme_tables.md"
    parts = [
        "<!-- generated by scripts/make_readme_tables.py -->",
        "\n## Embedding catalog\n",
        embedding_catalog(),
        "\n## db-RDA (marginal)\n",
        dbrda_table(),
        "\n## Social network\n",
        social_table(),
        "",
    ]
    out.write_text("\n".join(parts))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
