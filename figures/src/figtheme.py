"""One shared look for every figure (paper + rebuttal). Palette validated with the dataviz skill's validate_palette.py.
Output goes to figures/<name>.{svg,png}."""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'figures'

# categorical slots (fixed order): blue, orange, aqua, violet, yellow
BLUE, ORANGE, AQUA, VIOLET, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#eda100'
NAVY, SKY, RED, GREEN, MAGENTA = '#184f95', '#9ec5f4', '#e34948', '#008300', '#e87ba4'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e7e5df', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#f1f6fd', '#cde2fb', '#9ec5f4', '#5b9be8', '#2a78d6', '#184f95', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_blue_orange', ['#184f95', '#6da7ec', '#f4f3ef', '#f3a283', '#c9471b'])
CAT = [BLUE, ORANGE, AQUA, VIOLET, YELLOW]
DIFF3 = [SKY, BLUE, NAVY]          # ordinal easy < medium < hard: one hue, light to dark


def setup():
    mpl.rcParams.update({
        'font.family': 'Liberation Sans', 'mathtext.fontset': 'stixsans', 'svg.fonttype': 'path', 'font.size': 9.5,
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'text.color': INK,
        'axes.edgecolor': '#b9b8b0', 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.labelsize': 10,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8.8, 'ytick.labelsize': 8.8, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7, 'grid.linestyle': '-',
        'axes.axisbelow': True, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
        'legend.frameon': False, 'legend.fontsize': 8.6, 'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'errorbar.capsize': 0})


def save(fig, name, kind):
    d = FINAL if kind == 'paper' else FINAL / kind; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f'{name}.svg', bbox_inches='tight')
    fig.savefig(d / f'{name}.png', dpi=220, bbox_inches='tight'); plt.close(fig); print('wrote', kind, name)


def letter(ax, s, dx=-0.02, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='right', color=INK)


def dot(ax, x, y, c=BLUE, s=46, hollow=False, z=4, **kw):
    if hollow: return ax.scatter(x, y, s=s, facecolor=SURF, edgecolor=c, linewidth=1.7, zorder=z, **kw)
    return ax.scatter(x, y, s=s, color=c, edgecolor='white', linewidth=0.9, zorder=z, **kw)


def numbered(ax, x, y, colors, hollow=None, size=15):
    """Numbered markers (as in the paper), number in white; hollow markers carry a coloured number."""
    hollow = np.zeros(len(x), bool) if hollow is None else np.asarray(hollow)
    for i, (xi, yi, c, h) in enumerate(zip(x, y, colors, hollow), 1):
        ax.scatter([xi], [yi], s=size ** 2 * 0.62, facecolor=SURF if h else c, edgecolor=c if h else 'white', linewidth=1.6 if h else 0.9, zorder=4)
        ax.text(xi, yi, str(i), ha='center', va='center', fontsize=6.6, fontweight='bold', color=c if h else 'white', zorder=5)


def model_index(ax, names, title='Model index', cols=2, fs=7.4):
    ax.axis('off'); n = len(names); per = int(np.ceil(n / cols)); step = 0.92 / max(per, 11)
    ax.text(0, 1.0, title, fontsize=9, fontweight='bold', va='top', transform=ax.transAxes)
    for k, t in enumerate(names):
        j, r = divmod(k, per); ax.text(j * (1.0 / cols), 0.93 - r * step, f'{k + 1:>2}', fontsize=fs, va='top', transform=ax.transAxes, color=MUTED, family='DejaVu Sans')
        ax.text(j * (1.0 / cols) + 0.075 * (2 / cols), 0.93 - r * step, t, fontsize=fs, va='top', transform=ax.transAxes, color=INK2)


def p_text(r, p):
    ps = f'{p:.1e}'.replace('e-0', 'e−0').replace('e-', 'e−')
    return f'r = {r:.3f}, p = {ps}'


def legend_markers(ax, items, **kw):
    """items: [(label, color, marker, hollow)]"""
    hs = [Line2D([], [], marker=m, ls='', color=c, mfc=SURF if h else c, mec=c if (h or m == 'x') else 'white', mew=1.5 if (h or m == 'x') else 0.8, ms=7, label=l) for l, c, m, h in items]
    return ax.legend(handles=hs, **kw)


def heatmap(ax, M, cmap, vmin=0, vmax=1):
    """Vector heatmap (pcolormesh) with cell centres at integer coordinates, row 0 at the top -- same geometry as imshow."""
    M = np.ma.masked_invalid(np.asarray(M, float)); r, c = M.shape
    q = ax.pcolormesh(np.arange(c + 1) - 0.5, np.arange(r + 1) - 0.5, M, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat', edgecolor='face', linewidth=0.3)
    ax.set_xlim(-0.5, c - 0.5); ax.set_ylim(r - 0.5, -0.5); ax.grid(False); return q
