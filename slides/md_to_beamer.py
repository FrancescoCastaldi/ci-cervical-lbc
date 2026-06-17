#!/usr/bin/env python3
"""
Convert Computational_Imaging_Studio.md -> Beamer LaTeX (print-friendly, no images).
"""

import re, sys, os

MD = "../Computational_Imaging_Studio.md"
TEX = "slides/comp_imaging_beamer.tex"

# NOTE: { and } are NOT escaped here because we insert \textbf{...} etc.
# after escaping. Escaping braces would break those commands.
LATEX_SPECIAL = str.maketrans(
    {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "_": r"\_",
    }
)


def esc_text(t):
    """Escape special LaTeX chars (for non-math text)."""
    return t.translate(LATEX_SPECIAL)


UNICODE_MAP = {
    "\u2248": r"$\approx$",  # ≈
    "\u221a": r"$\surd$",  # √
    "\u2264": r"$\le$",  # ≤
    "\u2265": r"$\ge$",  # ≥
    "\u00b0": r"$^\circ$",  # °
    "\u03b1": r"$\alpha$",  # α
    "\u03b2": r"$\beta$",  # β
    "\u03bb": r"$\lambda$",  # λ
    "\u03c3": r"$\sigma$",  # σ
    "\u03b8": r"$\theta$",  # θ
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # Misc symbols, emoji, supplemental
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+"
)


def replace_unicode(text):
    """Replace Unicode mathematical symbols with LaTeX equivalents."""
    for uni, latex in UNICODE_MAP.items():
        text = text.replace(uni, latex)
    return text


def strip_emoji(text):
    """Remove emoji characters."""
    return EMOJI_PATTERN.sub("", text)


def fmt_inline(text):
    """Convert **bold** and *italic* markers. Protect math $...$ from escaping."""
    # Strip emoji first
    text = strip_emoji(text)
    # Replace unicode math symbols
    text = replace_unicode(text)

    parts = re.split(r"(\$[^$]*\$)", text)
    result = []
    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            result.append(part)  # Keep raw math
        else:
            # Bold
            part = re.sub(
                r"\*\*(.+?)\*\*", lambda m: r"\textbf{" + m.group(1) + r"}", part
            )
            # Italic (not **)
            part = re.sub(
                r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)",
                lambda m: r"\textit{" + m.group(1) + r"}",
                part,
            )
            result.append(esc_text(part))
    return "".join(result)


# Read & strip images
with open(MD, encoding="utf-8") as f:
    raw = f.read()
raw = re.sub(r"!\[.*?\]\(.*?\)", "", raw)
raw = re.sub(r"<img\s+[^>]*>", "", raw)
lines = raw.split("\n")

# ---- Write preamble ----
PRE = r"""\documentclass[italian,aspectratio=169]{beamer}
\usepackage[italian]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{mathtools}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}
% Metropolis theme
\usepackage{beamerthememetropolis}
\metroset{
  progressbar=frametitle,
  sectionpage=progressbar,
  numbering=counter,
  block=fill,
}

% Color palette
\definecolor{cb}{RGB}{41,128,185}   % Blue  - Intuizione
\definecolor{co}{RGB}{231,76,60}    % Red   - All'orale
\definecolor{cg}{RGB}{39,174,96}    % Green - Spiegazioni
\definecolor{cp}{RGB}{142,68,173}   % Purple - Termini

\setbeamercolor{background canvas}{bg=white!98}
\setbeamercolor{alerted text}{fg=co}
\setbeamercolor{example text}{fg=cg}
\setbeamercolor{normal text}{fg=black!90}

% Custom blocks (Metropolis-compatible, with fill)
\newenvironment{intuizione}{%
  \setbeamercolor{block title}{fg=white,bg=cb}%
  \setbeamercolor{block body}{fg=black,bg=cb!5!white}%
  \begin{block}{Intuizione}}{\end{block}}
\newenvironment{orale}{%
  \setbeamercolor{block title}{fg=white,bg=co}%
  \setbeamercolor{block body}{fg=black,bg=co!5!white}%
  \begin{block}{All'orale}}{\end{block}}
\newenvironment{spiegazione}{%
  \setbeamercolor{block title}{fg=white,bg=cg}%
  \setbeamercolor{block body}{fg=black,bg=cg!5!white}%
  \begin{block}{Spiegazioni}}{\end{block}}
\newenvironment{terminiblock}{%
  \setbeamercolor{block title}{fg=white,bg=cp}%
  \setbeamercolor{block body}{fg=black,bg=cp!5!white}%
  \begin{block}{Termini}}{\end{block}}

% Compact code blocks with shadow
\lstdefinestyle{pp}{
  language=Python,
  basicstyle=\ttfamily\small,
  backgroundcolor=\color{gray!3},
  frame=shadowbox,
  rulesepcolor=\color{gray!30},
  breaklines=true,
  showstringspaces=false,
  commentstyle=\color{gray},
  keywordstyle=\color{cb},
}

% Slightly smaller text for all frames (better density)
\apptocmd{\frame}{\small}{}{}

% Section pages (Metropolis built-in handles the rest)
\title{Computational Imaging}
\subtitle{Studio per l'Esame Orale}
\date{}

\begin{document}
\frame{\titlepage}
\frame{\frametitle{Indice}\tableofcontents}
"""

out = [PRE]

BLOCK_ENV_MAP = {
    "intuizione": "intuizione",
    "orale": "orale",
    "all'orale": "orale",
    "spiegazioni": "spiegazione",
    "termini": "terminiblock",
    "termini chiave": "terminiblock",
    "a cosa serve": None,  # handled as italic
}

in_code = False
frame_content = []
frame_title = None
section_title = None
accum_type = None  # 'block' or 'italic'
accum_lines = []


def flush_accum():
    global accum_lines, accum_type
    if not accum_lines:
        return
    t = "\n".join(accum_lines).strip()
    if accum_type == "block":
        frame_content.append(
            r"\begin{"
            + BLOCK_ENV_MAP.get(accum_label, "block")
            + "}"
            + fmt_inline(t)
            + r"\end{"
            + BLOCK_ENV_MAP.get(accum_label, "block")
            + "}"
        )
    elif accum_type == "italic":
        frame_content.append(r"\textit{" + fmt_inline(t) + "}")
    elif accum_type == "bold":
        frame_content.append(r"\textbf{" + fmt_inline(t) + "}")
    accum_lines = []
    accum_type = None


def wrap_items(lines):
    """Wrap consecutive \\item commands in itemize environment."""
    result = []
    in_list = False
    for line in lines:
        if line.startswith(r"\item "):
            if not in_list:
                result.append(r"\begin{itemize}")
                in_list = True
            result.append(line)
        else:
            if in_list:
                result.append(r"\end{itemize}")
                in_list = False
            result.append(line)
    if in_list:
        result.append(r"\end{itemize}")
    return result


def flush_frame():
    global frame_content, frame_title
    if not frame_content and not frame_title:
        return
    if frame_title is None:
        frame_title = ""
    # Check if content is too long -> allowframebreaks
    total_chars = sum(len(l) for l in frame_content)
    fb_opt = (
        "[allowframebreaks]" if (len(frame_content) > 20 or total_chars > 2000) else ""
    )
    out.append(r"\begin{frame}" + fb_opt + r"{" + frame_title + "}")
    processed = wrap_items(frame_content)
    for l in processed:
        out.append(l)
    out.append(r"\end{frame}")
    out.append("")
    frame_content = []
    frame_title = None


def add_section(title):
    flush_frame()
    out.append(r"\section{" + fmt_inline(title) + "}")
    out.append(r"\begin{frame}{" + fmt_inline(title) + "}")
    out.append(r"\tableofcontents[currentsection]")
    out.append(r"\end{frame}")
    out.append("")


# ---- Parse lines ----
for line in lines:
    s = line.strip()

    # Code blocks
    if s.startswith("```"):
        if in_code:
            out.append(r"\end{lstlisting}")
            in_code = False
        else:
            flush_accum()
            flush_frame()
            out.append(r"\begin{lstlisting}[style=pp]")
            in_code = True
        continue
    if in_code:
        out.append(line)
        continue

    # Images (should be gone)
    if re.match(r"^\s*!\[", s) or re.match(r"^\s*<img", s, re.I):
        continue

    # Headings
    hm = re.match(r"^(#+)\s+(.+)$", s)
    if hm:
        level = len(hm.group(1))
        title = re.sub(r"\s*`\[p\.[0-9\-]+\]`\s*", "", hm.group(2)).strip()
        flush_accum()
        if level == 1:
            flush_frame()
        elif level == 2:
            flush_frame()
            add_section(title)
        elif level == 3:
            flush_frame()
            frame_title = fmt_inline(title)
        elif level == 4:
            frame_content.append(r"\textbf{" + fmt_inline(title) + "}")
        continue

    # Empty line -> flush pending accum
    if not s:
        flush_accum()
        continue

    # Horizontal rule
    if s.startswith("---") and len(s) >= 3:
        flush_accum()
        frame_content.append(r"\medskip\noindent\rule{\textwidth}{0.4pt}\medskip")
        continue

    # Math display
    if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
        flush_accum()
        frame_content.append(r"\[ " + s[2:-2].strip() + r" \]")
        continue
    if s == "$$":
        flush_accum()
        frame_content.append(r"\[ ")
        continue

    # Blockquotes
    if s.startswith(">"):
        content = re.sub(r"^>\s?", "", s)
        # Handle special blockquote labels
        bqm = re.match(r"\*\*(.+?)\*\*:?\s*(.*)", content)
        if bqm:
            label = bqm.group(1).strip().lower()
            rest = bqm.group(2).strip()

            # Check if this label maps to a block environment
            env = None
            for key, val in BLOCK_ENV_MAP.items():
                if key in label:
                    env = val
                    break

            if env:
                flush_accum()
                if rest:
                    frame_content.append(
                        r"\begin{" + env + "}" + fmt_inline(rest) + r"\end{" + env + "}"
                    )
                else:
                    # Multi-line block - start accumulating
                    accum_type = "block"
                    accum_label = label
            elif env is None and ("cosa serve" in label):
                flush_accum()
                if rest:
                    frame_content.append(r"\textit{" + fmt_inline(rest) + "}")
                else:
                    accum_type = "italic"
            else:
                # Generic bold label
                flush_accum()
                text = rest if rest else ""
                frame_content.append(
                    r"\textbf{"
                    + fmt_inline(bqm.group(1).strip())
                    + "}: "
                    + fmt_inline(text)
                )
            continue

        # Continuation of a blockquote (already in accum)
        if accum_type:
            accum_lines.append(content)
            continue

        # Plain blockquote continuation (no special handling needed)
        frame_content.append(fmt_inline(content))
        continue

    # Unordered list
    if s.startswith("- "):
        flush_accum()
        frame_content.append(r"\item " + fmt_inline(s[2:]))
        continue
    if s.startswith("* ") and not s.startswith("**"):
        flush_accum()
        frame_content.append(r"\item " + fmt_inline(s[2:]))
        continue

    # Ordered list
    nm = re.match(r"^(\d+)\.\s+(.*)", s)
    if nm:
        flush_accum()
        frame_content.append(r"\item " + fmt_inline(nm.group(2)))
        continue

    # Regular paragraph
    flush_accum()
    frame_content.append(fmt_inline(s))

flush_accum()
flush_frame()
out.append(r"\end{document}")

with open(TEX, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("OK: " + TEX + " (" + str(len(out)) + " righe)")
