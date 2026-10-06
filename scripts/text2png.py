import sys

from matplotlib.figure import Figure


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: text2png.py OUT.png < text", file=sys.stderr)
        return 2
    text = sys.stdin.read().expandtabs(4).rstrip("\n") or " "
    lines = text.splitlines()
    width = max(len(line) for line in lines)
    fig = Figure(
        figsize=(max(2.0, 0.085 * width + 0.4), max(0.6, 0.19 * len(lines) + 0.3)),
        dpi=150,
    )
    fig.text(0.02, 0.98, text, family="monospace", fontsize=9, va="top", ha="left")
    fig.savefig(sys.argv[1], facecolor="white")
    return 0


if __name__ == "__main__":
    sys.exit(main())
