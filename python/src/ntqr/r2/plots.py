"""
Plotting functions for binary tests (R=2).

Functions:
    plot_evaluations

Misc variables:

"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from adjustText import adjust_text


def prepare_accuracy_axes(axes, title):
    axes.set(title=title, xlim=(0, 100), ylim=(0, 100))

    for dax in (axes.xaxis, axes.yaxis):
        dax.set(
            major_locator=ticker.FixedLocator([50]),
            major_formatter=ticker.FixedFormatter(["50"]),
            minor_locator=ticker.FixedLocator(
                [0, 10, 20, 30, 40, 60, 70, 80, 90, 100]
            ),
            minor_formatter=ticker.FixedFormatter(
                ["0", "10", "20", "30", "40", "60", "70", "80", "90", "100"]
            ),
        )

    # Styling them
    axes.tick_params(
        axis="both", which="major", grid_color="g", grid_linewidth=2
    )
    axes.tick_params(
        axis="both", which="minor", grid_color="k", grid_alpha=0.4
    )

    # Blindspot line
    axes.plot([0, 100], [100, 0], color="r")

    # The labels for the four quadrants of the unit cube
    quad_label_info = [
        ("A", (75, 75)),
        ("C", (25, 25)),
        ("B", (25, 75)),
        ("D", (75, 25)),
    ]
    for quad_label, pos in quad_label_info:
        axes.annotate(quad_label, pos, size=18, alpha=0.4)

    axes.set_aspect("equal", "box")

    return


def _plot_evaluations(ax, evals, title="Title", legend_loc="best"):
    prepare_accuracy_axes(ax, title)

    # Specifying the grid
    ax.grid("on")

    texts_to_adjust = []
    for label, x, y, marker, colors, sizes in evals:
        ax.scatter(x, y, s=sizes, c=colors, marker=marker, label=label)
        # Adding the classifier label to each point
        for i in range(len(x)):
            texts_to_adjust.append(ax.annotate(str(i + 1), (x[i], y[i])))

    ax.legend(loc=legend_loc)
    # adjust_text(texts_to_adjust)


def compare_evaluations(
    evals, gt_index, titles=[], figsize=(10, 5), legend_loc="best"
):
    fig, axs = plt.subplots(
        1, len(evals) - 1, figsize=figsize, layout="constrained"
    )

    j = 0
    for i in range(len(evals)):
        if i == gt_index:
            continue

        _plot_evaluations(
            axs[j],
            (evals[gt_index], evals[i]),
            title=titles[i],
            legend_loc=legend_loc,
        )
        j += 1

    plt.show()

    return


def plot_evaluations(
    evals, title="Title", figsize=(5, 4), legend_loc="lower left"
):
    plt.style.use("_mpl-gallery")

    # plot
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    _plot_evaluations(ax, evals, title=title, legend_loc=legend_loc)

    plt.show()

    return
