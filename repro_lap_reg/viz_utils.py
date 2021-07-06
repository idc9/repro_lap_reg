import os
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns


def savefig(fpath, dpi=100, close=True):
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)
    if close:
        plt.close()


def maybe_save(name, save_dir=None, dpi=100, close=True):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        savefig(join(save_dir, name), dpi=dpi, close=True)


def hierarchical_color_palette(cat_subcats,
                               cat_palette="tab10", light=True):
    """
    Gets a hierachical color palette for categories/subcategories.

    Parameters
    ----------
    cat_subcats: dict of lists
        List of subcategories for each catgory. Must be a dict where the keys are the categories. Each value must be a list of the corresponding subcategory names. All subcategory names must be unique.

    base_palette: str
        Base palette for getting the colors for each category

    light: bool
        Use sns.light_palette or sns.dark_palette

    Output
    ------
    subcat_colors, cat_colors

    subcat_colors: dict
        Colors for each sub-category.

    cat_colors: dict
        Colors for each category.

    """
    assert type(cat_subcats) == dict

    # each entry of cat_subcats should be a list
    assert all(hasattr(v, '__len__') for v in cat_subcats.values())

    # make sure all subcat names are uniuqe
    # assert all_unique(sc for sc in subcats for subcats in cat_subcats.values())

    cats = list(cat_subcats.keys())
    n_cats = len(cats)

    # base colors for categories
    cat_colors = sns.color_palette(palette=cat_palette,
                                   n_colors=n_cats)
    cat_colors = {cats[i]: color for i, color in enumerate(cat_colors)}

    if light:
        pal = sns.light_palette
    else:
        pal = sns.dark_palette

    # make colors for each subcategory
    subcat_colors = {}
    for cat in cats:
        subcats = cat_subcats[cat]

        # create palette for each subcategory
        subcat_pal = pal(color=cat_colors[cat],
                         n_colors=len(subcats) + 1,
                         reverse=True)[:-1]

        # get color foe each sub cat in this cat
        for sc_idx, sc in enumerate(subcats):
            subcat_colors[sc] = subcat_pal[sc_idx]

    return subcat_colors, cat_colors
