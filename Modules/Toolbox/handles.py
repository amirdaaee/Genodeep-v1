import numpy as np
import pandas as pd
from IPython import display
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import os


def canseek_load(path, dataset_main_name=None, folds=None, load_pnas=True, load_ddef=True):
    """ load cancer-seek dataset

    Parameters
    ----------
    path : str
        path to Data repo
    dataset_main_name : str , optional
        name of dataset_main dataset file
        default to 'dataset_main_dataset.xlsx'
    folds : list[list[int]] , optional
        default to load all as single dataset
    load_pnas : bool | str
        whether to load PNAS dataset
        if str, loads path+load_pnas file
        otherwise loads loads path+'pnas_dataset.csv' file
    load_ddef : bool | str
            whether to load data dictionary
            if str, loads path+load_ddef file
            otherwise loads loads path+'dataset_main_ddef.csv' file

    Returns
    -------
    (list[pd.DataFrame] , pd.DataFrame | None,pd.DataFrame | None)

    """
    if dataset_main_name is None:
        dataset_main_name = 'dataset_main_dataset.xlsx'
    canseek_path = os.path.join(path, dataset_main_name)
    xl = pd.ExcelFile(canseek_path)
    if folds is None:
        folds = [[i for i in range(len(xl.sheet_names))]]
    dts = []
    for i in folds:
        dt = pd.DataFrame()
        for j in i:
            dt = pd.concat([dt, xl.parse(j, index_col=0)])
        dts.append(dt)

    pnas, ddef = None, None
    if load_pnas:
        if load_pnas.__class__ is str:
            pnas_path = os.path.join(path, load_pnas)
        else:
            pnas_path = os.path.join(path, 'pnas_dataset.csv')
        if pnas_path.endswith('.csv'):
            loader = pd.read_csv
        else:
            loader = pd.read_excel
        pnas = loader(pnas_path, index_col=0)

    if load_ddef:
        if load_ddef.__class__ is str:
            ddef_path = os.path.join(path, load_ddef)
        else:
            ddef_path = os.path.join(path, 'dataset_main_ddef.csv')
        if ddef_path.endswith('.csv'):
            loader = pd.read_csv
        else:
            loader = pd.read_excel
        ddef = loader(ddef_path, index_col=0)

    return (dts, pnas, ddef)


def name_to_alias(names, ddef):
    """ generates alias names of features

    Parameters
    ----------
    names : list[str]
    ddef : pd.DataFrame
        index : alias
        should include name column

    Returns
    -------
    list[str]
    """

    ddef = ddef.copy()
    if 'Alias' not in ddef.columns:
        ddef['Alias'] = ddef.index
    ddef.set_index('Short_Name', inplace=True)
    aliases = []
    for n in names:
        aliases.append(ddef.loc[n, 'Alias'])
    return aliases


def col_selector(cols, include_col=None, exclude_col=None):
    """get requested columns between all columns

    Parameters
    ----------
    cols : pd.DataFrame.columns or np.array or list -> 1D
        list of all columns
    include_col , exclude_col : list or None
        can not being specified together
        both None -> all columns

    Returns
    ----------
    np.ndarray -> 1D
        selected columns
    """

    if not cols.__class__ == np.ndarray:
        cols = np.array(cols)

    assert cols.ndim == 1, 'cols parameter must be 1D'

    assert (include_col is None) or (exclude_col is None), 'both include and exclude can not be set'

    if (include_col is None) and (exclude_col is None):
        pass

    elif exclude_col is None:
        cols = include_col

    elif include_col is None:
        cols = cols[~np.isin(cols, exclude_col)]

    return cols


def strat_folder(dataset, strt_keys, n_folds=10, report=False):
    """generate stratified folds

    Parameters
    ----------
    dataset : pd.DataFrame
        original dataset
    strt_keys : list
        column names to be stratified perceived
    n_folds : int > 0
        number of required folds
    report : bool
        whether display fold values report or not

    Returns
    ----------
    list of lists of lists
        [[Kfold1 train idxs,Kfold1 test idxs],...]
    """

    dataset = dataset.copy()
    keys = dataset[strt_keys].astype(str).values.sum(axis=1)
    keys = pd.DataFrame(keys, index=dataset.index)
    dataset['keys'] = keys
    # for c, k in enumerate(keys.unique()):
    #     keys.loc[keys == k] = c
    x = keys.index.values
    keys = keys.values
    folds = []

    rep = None
    if report:
        rep = pd.DataFrame(columns=np.unique(keys))
    c = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for train, test in skf.split(x, keys):
        folds.append([x[train], x[test]])
        if rep is not None:
            tmp_rep = dataset.loc[x[test], 'keys'].value_counts(sort=False)
            rep = rep.append(tmp_rep, ignore_index=True)
            c += 1
    if rep is not None:
        rep = rep.fillna(0).astype(int)
        ratio = rep.transpose() / rep.sum(axis=1).values * 100
        ratio = ratio.applymap(lambda xx: np.round(xx, 1))
        rep.index = ['{}(n)'.format(x) for x in rep.index]
        ratio.columns = ['{}(%)'.format(x) for x in ratio.columns]
        print('Kfolds report:')
        display.display(rep.transpose())
        display.display(ratio)
        print('-' * 20)

    return folds


def subset_selector(dataset, sets, sets_cols):
    """ get a subset of dataset the set_col column value is in sets

    Parameters
    ----------
    dataset : pd.DataFrame
    sets : list(str)
    sets_cols : str

    Returns
    -------
    pd.DataFrame
    """

    return dataset.loc[dataset[sets_cols].isin(sets)].copy()


def dataset_generator(dataset, columns):
    """ generates np.ndarray dataset from pd.DataFrame

    Parameters
    ----------
    dataset : pd.DataFrame
    columns : list(str) | list(list(str))

    Returns
    -------
    np.ndarray | list(np.ndarray)
    """
    nested = False
    if columns[0].__class__ == list:
        nested = True
    if nested:
        ret = []
        for i in columns:
            ret.append(dataset[i].values)
    else:
        ret = dataset[columns].values

    return ret


def print_full(ds):
    pd.set_option('display.max_columns', ds.shape[1])
    display.display(ds)
    pd.reset_option('display.max_columns')


def default_fig(dpi=True, figsize=False):
    defs = {'dpi': 150, 'figsize': (15, 8)}
    inps = {'dpi': dpi, 'figsize': figsize}
    figargs = {}
    for arg in defs.keys():
        if inps[arg].__class__ is bool:
            if inps[arg]:
                figargs[arg] = defs[arg]
            else:
                figargs[arg] = None
        else:
            figargs[arg] = inps[arg]
    fig = plt.figure(**figargs)
    return fig
