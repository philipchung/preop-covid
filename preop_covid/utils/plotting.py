import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_count_percent_plot(
    data: pd.DataFrame,
    x: str,
    hue: str,
    xlabel: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (6, 10),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Generates side-by-side count bar plot & percentage area plot.

    Args:
        data (pd.DataFrame): examples to consider for plot
        x (str): feature used to generate counts and percentages
        hue (str): feature by which we stratify the counts and percentages
        xlabel (str | None, optional): Optional xlabel on plots.
        title (str | None, optional): Optional overall title for figure.
        figsize (tuple[int, int], optional): Figure size (height, width). Defaults to (6, 10).

    Returns:
        tuple[plt.Figure, list[plt.Axes]]: Tuple of figure and list of axes.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # Count Plot
    sns.histplot(
        data=data,
        x=x,
        hue=hue,
        stat="count",
        multiple="dodge",
        ax=ax[0],
    )
    ax[0].set(
        title=f"{hue}, Case Counts",
        xlabel=x if xlabel is None else xlabel,
    )
    for container in ax[0].containers:
        ax[0].bar_label(container, label_type="edge", fmt="%g")
    # Percent Plot
    sns.histplot(
        data=data,
        x=x,
        hue=hue,
        stat="percent",
        multiple="fill",
        ax=ax[1],
    )
    ax[1].set(
        title=f"{hue}, Percentage of Cases",
        xlabel=x if xlabel is None else xlabel,
    )
    for container in ax[1].containers:
        ax[1].bar_label(container, label_type="center", fmt="%.2f")
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    return fig, ax


def make_count_percent_plots(
    data: pd.DataFrame,
    x: str,
    hue: str | list[str],
    xlabel: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Generates side-by-side count bar plot & percentage area plots with the ability to
    multiplex plots for multiple features in `hue`.

    Args:
        data (pd.DataFrame): examples to consider for plot
        x (str): feature used to generate counts and percentages
        hue (str | list[str]): feature(s) by which we stratify the counts and percentages.
        xlabel (str | None, optional): Optional xlabel on plots.
        title (str | None, optional): Optional overall title for figure.
        figsize (tuple[int, int], optional): Figure size (height, width). Defaults to (6, 10).

    Returns:
        tuple[plt.Figure, list[plt.Axes]]: Tuple of figure and list of axes.
    """
    if isinstance(hue, str):
        figsize = (10, 6) if figsize is None else figsize
        return make_count_percent_plot(
            data=data, x=x, hue=hue, xlabel=xlabel, title=title, figsize=figsize
        )
    else:
        figsize = (10, 12) if figsize is None else figsize
        num_hues = len(hue)
        fig, ax = plt.subplots(nrows=num_hues, ncols=2, figsize=figsize)

        for idx, h in enumerate(hue):
            ct_ax = ax[idx, 0]
            pct_ax = ax[idx, 1]
            # Count Plot
            sns.histplot(
                data=data,
                x=x,
                hue=h,
                stat="count",
                multiple="dodge",
                ax=ct_ax,
            )
            ct_ax.set(
                title=f"{h}, Case Counts",
                xlabel=x if xlabel is None else xlabel,
            )
            for container in ct_ax.containers:
                ct_ax.bar_label(container, label_type="edge", fmt="%g")
            # Percent Plot
            sns.histplot(
                data=data,
                x=x,
                hue=h,
                stat="percent",
                multiple="fill",
                ax=pct_ax,
            )
            pct_ax.set(
                title=f"{h}, Percentage of Cases",
                xlabel=x if xlabel is None else xlabel,
            )
            for container in pct_ax.containers:
                pct_ax.bar_label(container, label_type="center", fmt="%.2f")
        if title is not None:
            plt.suptitle(title)
            plt.tight_layout(rect=[0, 0.03, 1, 0.99])
        else:
            plt.tight_layout()
    return fig, ax
