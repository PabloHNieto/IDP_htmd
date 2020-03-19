def hide_spines(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)

def annotate_axes(fig):
  for i, ax in enumerate(fig.axes):
      ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
      ax.tick_params(labelbottom=False, labelleft=False)
        
def change_spines_color(ax, color, lw):
  ax.spines['top'].set_color(color)
  ax.spines['right'].set_color(color)
  ax.spines['bottom'].set_color(color)
  ax.spines['left'].set_color(color)
  ax.spines['top'].set_linewidth(lw)
  ax.spines['right'].set_linewidth(lw)
  ax.spines['bottom'].set_linewidth(lw)
  ax.spines['left'].set_linewidth(lw)