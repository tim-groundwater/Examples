"""
Created on Tue Mar 14 13:37:55 2023

@author: HMEUW
"""

import numpy as np
import pandas as pd
import timml as tml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict


def create_xsection_model(df_model_input, scen, verbose=False):
    """Create xsection model around a leveel using TimML ModelMaq model.

    Some remarks
    - steady state calculation
    - dike itself not included in groundwater flow model
    - check on uplift not included
    - piping not included

    Parameters
    ----------
    df_model_input : pd.DataFrame
        Model parameters, created by import_model_parameters
    scen : string
        column df_model_input to use

    Returns
    -------
    ml : timml.model.ModelMaq
        The groundwater model.
    """
    # create model layers from input lists in dataframe
    kh, z, c, npor, litho = input_to_model_layers(
        df_model_input, scen=scen, top_type='semi', verbose=verbose
    )
    if verbose:
        print("kh", len(kh), kh)
        print("z", len(z), z)
        print("c", len(c), c)
        print("npor", len(npor), npor)
        print("litho", len(litho), litho)

    # create model itself
    ml = tml.ModelMaq(
        kaq=kh,
        z=z[1:], # skip item 0 because model top condition is confined
        c=c[1:],  # skip item 0 because model top condition is confined
        npor=npor[1:],  # skip item 0 because model top condition is confined
        topboundary="conf",
    )
    ml.name = scen

    
    # add boundary conditions
    # boundary condition in river summer bed/channel
    # change kh and c from aquifer to surface water
    # very high kh and very low c value, porosity is 1
    kh_surfacewater = 1000
    c_surfacewater = 1e-10
    npor_surface = 1
    kh_inhom = np.asarray(kh.copy())
    kh_inhom[
        np.where(ml.aq.zaqtop > df_model_input.loc["channel_bottom", scen])
    ] = kh_surfacewater
    c_inhom = np.asarray(c.copy())
    c_inhom[
        np.where(ml.aq.zaqtop > df_model_input.loc["channel_bottom", scen])
    ] = c_surfacewater

    
    if df_model_input.loc["channel_resistance", scen] > 0:
        # has the channel resistance? yes, 
        # add channel_resistance to c_inhom
        c_inhom[
            np.where(ml.aq.zaqtop > df_model_input.loc["channel_bottom", scen])
        ] = df_model_input.loc["channel_resistance", scen]
        # add horizontal channel resistance via lower kh at interface
        x_channel_resistance = 0.1
        x_channel_min_model = (
            df_model_input.loc["x_channel_min", scen] - x_channel_resistance
        )
    else:
        # no action for c_inhom and horizontal channel resistance required
        x_channel_min_model = df_model_input.loc["x_channel_min", scen]

    # porosity deals with aquifers and aquitards, calculate max aquifer id first
    npor_inhom_max_aq = np.where(
        ml.aq.zaqtop > df_model_input.loc["channel_bottom", scen]
    )[0][-1]
    npor_inhom = np.asarray(npor.copy())
    npor_inhom[np.arange((npor_inhom_max_aq * 2) + 1)] = npor_surface

    # add river summer bed/channel to model
    tml.StripInhomMaq(
        ml,
        x1=-np.inf,
        x2=x_channel_min_model,
        kaq=kh_inhom,
        z=z,
        c=c_inhom,
        npor=npor_inhom,
        topboundary="semi",
        hstar=df_model_input.loc["h_channel", scen],
        # label="channel" # label (not yet) supported in TimML
    )

    # add resistance for horizontal flow from surface water
    if x_channel_min_model != df_model_input.loc["x_channel_min", scen]:
        # update kaq within layers where hydraulic load is
        # calculate kh that is equivalent to channel_resistance
        kh_value_channel_resistance = (
            x_channel_resistance / df_model_input.loc["channel_resistance", scen]
        )
        kh_channel_resistance = kh_inhom.copy()
        kh_channel_resistance[
            kh_channel_resistance == kh_surfacewater
        ] = kh_value_channel_resistance

        tml.StripInhomMaq(
            ml,
            x1=x_channel_min_model,
            x2=df_model_input.loc["x_channel_min", scen],
            kaq=kh_channel_resistance,
            z=z,
            c=c,
            npor=npor,
            topboundary="semi",
            hstar=df_model_input.loc["h_channel", scen],
            # label="horizontal channel resistance" # label (not yet) supported in TimML
        )

    # add boundary condition on foreshore
    if (
        df_model_input.loc["x_outer_toe", scen]
        == df_model_input.loc["x_channel_min", scen]
    ):
        # outer toe has same x as channel
        # move outer toe one meter to crest, because it is easy as all models have a foreshore
        x_foreland_min = df_model_input.loc["x_outer_toe", scen] + 1
    else:
        x_foreland_min = df_model_input.loc["x_outer_toe", scen]
    # layer properties are same as model
    tml.StripInhomMaq(
        ml,
        x1=df_model_input.loc["x_channel_min", scen],
        x2=x_foreland_min,
        kaq=kh,
        z=z,
        c=c,
        npor=npor,
        topboundary="semi",
        hstar=df_model_input.loc["h_channel", scen],
        # label="foreshore" # label (not yet) supported in TimML
    )

    # add boundary condition in levee
    # add extra resistance, and average of level river and hinterland
    c_levee = np.asarray(c.copy())
    if c_levee[0] < 1:
        c_levee[0] = 1000
    else:
        c_levee[0] *= 1000
    h_levee = np.mean(
        [
            df_model_input.loc["h_topsystem", scen],
            df_model_input.loc["h_channel", scen],
        ]
    )
    tml.StripInhomMaq(
        ml,
        x1=df_model_input.loc["x_outer_toe", scen],
        x2=df_model_input.loc["x_inner_toe", scen],
        kaq=kh,
        z=z,
        c=c_levee,
        npor=npor,
        topboundary="semi",
        hstar=h_levee,
        # label="levee" # label (not yet) supported in TimML
    )
    
    # add hinterland strip and constant far away
    tml.StripInhomMaq(
        ml,
        x1=df_model_input.loc["x_inner_toe", scen],
        x2=np.inf,
        kaq=kh,
        z=z[1:], # skip item 0 because model top condition is confined
        c=c[1:], # skip item 0 because model top condition is confined
        npor=npor[1:], # skip item 0 because model top condition is confined
        topboundary="conf",
        # label="hinterland" # label (not yet) supported in TimML
    )
    tml.Constant(ml, xr=df_model_input.loc["x_hinterland_max", scen],
                 yr=0, hr=df_model_input.loc["h_aquifer_right_model", scen],
                 label='constant_top_aquifer_id0')

    # boundary condition in teensloot
    xlsv_ids = ["x_first_ditch_min", "x_first_ditch_max", "n_elements_ditch"]
    hls_id = "h_ditch"
    res_id = "resistance_ditch"
    layers = np.where(ml.aq.zaqtop > df_model_input.loc["bottom_ditch", scen])[0]
    add_hls(
        df_model_input,
        scen,
        ml,
        xlsv_ids,
        hls_id,
        res_id,
        layers=layers,
        label="first_ditch",
    )

    # ditches after first ditch
    for nrditch in range(1, df_model_input.loc["nr_ditches_hinterland", scen]):
        x_new_ditch = (
            df_model_input.loc["x_first_ditch_max", scen]
            + nrditch * df_model_input.loc["distance_between_ditches", scen]
        )
        layers = np.where(ml.aq.zaqtop > df_model_input.loc["bottom_ditch", scen])[0]
        if verbose:
            print("new ditch:", x_new_ditch, "| layers ", layers)
        tml.HeadLineSink1D(
            ml,
            xls=x_new_ditch,
            hls=df_model_input.loc["h_ditch", scen],
            res=df_model_input.loc["resistance_ditch", scen],
            wh=1,
            layers=layers,
            label=f"hinterland_id{nrditch}",
        )

    # solve
    ml.solve(silent=~verbose)

    return ml


def import_model_parameters(fn_xls, sheet_name, keepcols=None):
    """Import model parameter from Excel and does some pre-processing.

    Parameters
    ----------
    fn_xls : string
        filename of Excel file that has import data.
    sheet_name : string
        sheet_name in Excel file.
    keepcols : list, optional
        column names to keep. The default is None.

    Returns
    -------
    df_data : pd.DataFrame
        Model parameters, is input for create_xsection_model.

    """
    # import data from Excel to DataFrame
    df_data = pd.read_excel(
        fn_xls,
        sheet_name=sheet_name,
        skiprows=0,
    )
    # drop columns and set index
    df_data.drop("category", axis=1, inplace=True)
    df_data.set_index("parameter", inplace=True)

    # remove cols?
    if keepcols is not None:
        # keep only requested columns
        if not isinstance(keepcols, list):
            keepcols = [keepcols]
        df_data = df_data[keepcols]
    else:
        keepcols = df_data.columns[1:]

    # add row to dataframe
    # df_new_row = pd.DataFrame([[None] * len(df_data.columns)], columns=df_data.columns)
    # df_new_row["new_index"] = "x_hinterland_max"
    # df_new_row.set_index("new_index", inplace=True)
    # df_data = pd.concat([df_data, df_new_row])

    # for col in df_data.columns:
    #     if col not in ["parameter", "unit", "description"]:
    #         x0 = df_data.loc["x_first_ditch_max", col]
    #         nrditch = df_data.loc["nr_ditches_hinterland", col] - 0.5
    #         L = df_data.loc["distance_between_ditches", col]
    #         df_data.loc["x_hinterland_max", col] = x0 + nrditch * L

    # loop over rows
    for index, row in df_data[keepcols].iterrows():
        for col in keepcols:
            # only process cols to keep
            if col not in ["remark"]:
                # do not change remark col

                # convert list input to lists
                if str(row[col]).startswith("list:"):
                    # convert list values to list
                    input = row[col][5:].split("|")
                    export = []
                    for item in input:
                        # DISCUSS: I have to use a nested list in order to assign list to DataFrame. Why?
                        # see 'hacky' in https://stackoverflow.com/questions/26483254/python-pandas-insert-list-into-a-cell
                        # what is a better solution
                        export.append([float(item)])

                    df_data.loc[index, col] = export

    # add plotting assistance

    return df_data


def add_hls(
    df_model_input,
    scen,
    ml,
    xlsv_ids,
    hls_id,
    res_id,
    location_elements="left-bottom-right",
    layers=0,
    y=0,
    label=None,
):
    """Generate HeadLineSink1D for TimML model"""

    # update input
    if (not isinstance(layers, np.ndarray)) & (not isinstance(layers, list)):
        layers = np.asarray([layers])
    if isinstance(res_id, str):
        res = df_model_input.loc[res_id, scen]
    else:
        res = res_id

    # x locations
    xmin = df_model_input.loc[xlsv_ids[0], scen]
    xmax = df_model_input.loc[xlsv_ids[1], scen]
    n_elem = df_model_input.loc[xlsv_ids[2], scen]
    # apply linspace?
    if n_elem == 1:
        xlsv = [xmin, xmax]
    else:
        xlsv = np.linspace(xmin, xmax, n_elem)

    for i, x in enumerate(xlsv):
        # choose layers to apply
        if x == xlsv[0]:
            # left border
            if ("left" in location_elements) or ("all" in location_elements):
                layers_this_x = layers
            else:
                layers_this_x = layers[-1]
        elif x == xlsv[-1]:
            # right border
            if ("right" in location_elements) or ("all" in location_elements):
                layers_this_x = layers
            else:
                layers_this_x = layers[-1]
        else:
            if "all" in location_elements:
                layers_this_x = layers
            else:
                layers_this_x = layers[-1]

        if label is None:
            label = xlsv_ids[0]

        ls = tml.HeadLineSink1D(
            ml,
            xls=x,
            hls=df_model_input.loc[hls_id, scen],
            res=res,
            wh=1,
            layers=layers_this_x,
            label=f"{label}_id{i}",
        )

    return ls


def create_plot_model(df_model_input, scen, dh=0.5, close_plot=False, verbose=False):
    """
    Create xsection model and make some plots


    Parameters
    ----------
    df_model_input : pd.Dataframe
        Model input
    scen : str
        Colname in df_model_input.
    dh : float, optional
        Interval in contour plot. The default is 0.5.
    close_plot : boolean, optional
        Close created plots. The default is False.
    verbose : boolean, optional
        Print some info. The default is False.

    Returns
    -------
    ml : timml model
        Model.
    df_ml_layer : pd.Dataframe
        Model layers.
    name_per_bc : list
        Name in discharge_per_bc.
    discharge_per_bc : lst
        Discharge per boundary condition and at referenceline (x=0).
    first_layer_per_aquifer : list
        id of each first layer of an aquifer.
    last_layer_per_aquifer : list
        id of each last layer of an aquifer.

    """

    # create model
    ml = create_xsection_model(df_model_input, scen, verbose=verbose)

    # plot contour
    plot_contour(ml, df_model_input, scen=scen, dh=dh)

    # h and disvec
    fig, ax, all_qx = plot_q_and_disvec(ml, df_model_input, scen)

    plot_h(ml, df_model_input, scen)

    if close_plot:
        plt.close("all")

    name_per_bc, discharge_per_bc = calculate_q_semi(ml, verbose=verbose)

    names_discharge = name_per_bc
    discharge_per_location = discharge_per_bc

    names_discharge.append("reflijn")
    discharge_per_location.append(all_qx[2])

    first_layer_per_aquifer, last_layer_per_aquifer = get_layers_per_aquifer(ml)

    # verhouding
    qx_reflijn_wvp1 = all_qx[2][
        first_layer_per_aquifer[1] : last_layer_per_aquifer[1]
    ].sum()
    qx_reflijn_wvp2 = all_qx[2][
        first_layer_per_aquifer[2] : last_layer_per_aquifer[2]
    ].sum()

    if verbose:
        print(
            f"Q1 {qx_reflijn_wvp1:0.1f}; Q2 {qx_reflijn_wvp2:0.1f}; Q tot {qx_reflijn_wvp1+qx_reflijn_wvp2:0.1f}"
        )
        print(f"Q1 tov Q2 {(qx_reflijn_wvp1/qx_reflijn_wvp2)*100:0.1f} (%)")

    return (
        ml,
        names_discharge,
        discharge_per_location,
        first_layer_per_aquifer,
        last_layer_per_aquifer,
    )


def input_to_model_layers(df_model_input, scen, top_type='conf', verbose=False):
    """Create model layer parameters based on input lists.

    Parameters
    ----------
    df_model_input : pd.DataFrame
        DataFrame that holds lists with parameters per model layer
    scen : string
        column df_model_input to use

    Returns
    -------
    kh : lst
        kh for all aquifer layers.
    z : TYPE
        z for both aquifer and aquitard layers.
    c : TYPE
        c for all aquitard layers.
    npor : TYPE
        npor for both aquifer and aquitard layers.
    litho : TYPE
        litholgy string for both aquifer and aquitard layers.

    TO DO:
        - create nice dataframe for aquifer and aquitard layers

    """
    # create empty lists
    kh = []
    kzoverkh = []
    z = []
    c = []
    npor = []
    litho = []

    # add top of first layer
    z.append(df_model_input.loc["z_groundlevel", scen])

    # loop over all aquifers
    for aq_id in range(len(df_model_input.loc["bot_per_aquifer", scen])):
        if verbose:
            print(f"process aquifer {aq_id}")

        # create bottom within this aquifer
        # first do some calculation
        bot_this_aquifer = df_model_input.loc["bot_per_aquifer", scen][aq_id][0]
        nlay_this_aquifer = int(
            df_model_input.loc["nsublayers_per_aquifer", scen][aq_id][0]
        )

        # use np.linspace
        ztop_and_bots_this_aquifer = np.linspace(
            z[-1], bot_this_aquifer, nlay_this_aquifer + 1
        )
        if len(ztop_and_bots_this_aquifer) == 1:
            # applies for thin layers
            # np.linspace will only return top when nlay = 1
            ztop_and_bots_this_aquifer = [z[-1], bot_this_aquifer]

        # layer thinckness
        
        # ztop of the aquifer is included in previous aquifer (or z model top)
        bots_this_aquifer = ztop_and_bots_this_aquifer[1:]

        if verbose:
            print(
                f"top {z[-1]}; bot_this_aquifer:{bot_this_aquifer}, nlay_this_aquifer {nlay_this_aquifer}\nbots_this_aquifer {bots_this_aquifer}"
            )

        # loop over bottoms within this aquifer
        for i, bot in enumerate(bots_this_aquifer):
            # create an aquitard and aquifer part

            # add aquitard part
            z.append(
                z[-1]
                - df_model_input.loc[
                    "lay_thick_aquitard_part_of_modellayer_per_aquifer", scen
                ]
            )
            npor.append(df_model_input.loc["npor_per_aquifer", scen][aq_id][0])
            litho.append(f"aquitard part {i} of aq{aq_id}")

            # add aquifer part
            z.append(bot)
            kh.append(df_model_input.loc["kh_per_aquifer", scen][aq_id][0])
            kzoverkh.append(
                df_model_input.loc["kzoverkh_per_aquifer", scen][aq_id][0]
            )
            npor.append(df_model_input.loc["npor_per_aquifer", scen][aq_id][0])
            litho.append(f"aquifer part {i} of aq{aq_id}")

    # calculate kv for each aquitard
    # kh is list for each aquifer, already available
    # kv is to be calculated for each aquitard

    for lay_aquitard in range(len(kh)):
        if verbose:
            print(f"KH KV {lay_aquitard}")
        # resistance from layer that is on top of this layer
        if lay_aquitard > 0:
            # only when layer is not first layer
            top_aquifer_above = z[-2 + lay_aquitard * 2]
            bot_aquifer_above = z[0 + lay_aquitard * 2]
            d_aquifer_above = top_aquifer_above - bot_aquifer_above
            kh_aquifer_above = kh[lay_aquitard - 1]
            kzoverkh_aquifer_above = kzoverkh[lay_aquitard - 1]
            kv_aquifer_above = kh_aquifer_above / kzoverkh_aquifer_above
            c_above = (0.5 * d_aquifer_above) / kv_aquifer_above
            if verbose:
                print(
                    "above",
                    top_aquifer_above,
                    bot_aquifer_above,
                    d_aquifer_above,
                    kh_aquifer_above,
                    kzoverkh_aquifer_above,
                    kv_aquifer_above,
                    c_above,
                )
        else:
            c_above = 0

        # resistance from layer that is below of this layer
        top_aquifer_below = z[0 + lay_aquitard * 2]
        bot_aquifer_below = z[2 + lay_aquitard * 2]
        d_aquifer_below = top_aquifer_below - bot_aquifer_below
        kh_aquifer_below = kh[lay_aquitard + 0]
        kzoverkh_aquifer_below = kzoverkh[lay_aquitard + 0]
        kv_aquifer_below = kh_aquifer_below / kzoverkh_aquifer_below
        c_below = (0.5 * d_aquifer_below) / kv_aquifer_below
        if verbose:
            print(
                "below",
                top_aquifer_below,
                bot_aquifer_below,
                d_aquifer_below,
                kh_aquifer_below,
                kzoverkh_aquifer_below,
                kv_aquifer_below,
                c_below,
            )
            
        if verbose:
            print("total", c_above + c_below)

        c.append(c_above + c_below)

    if top_type == 'conf':
        # skip first item
        z=z[1:]
        c=c[1:],
        npor=npor[1:],
    
    return kh, z, c, npor, litho


def plot_contour(ml, df_model_input, scen, figsize=(12, 4), dh=0.5, ylim=None):
    """
    Create a contour plot of xsection model and boundary conditions

    Parameters
    ----------
    ml : timml.model.ModelMaq
        The groundwater model.
    channel : dictonary
        Model parameters regarding river conditions.
    df_model_input : pd.DataFrame
        Table of model input.
    figsize : tuple, optional
        size of figure. The default is (12,5).
    dh : float, optional
        interval in contours. The default is 0.5.

    Returns
    -------
    vc_plot : ?
        TODO: not working yet.

    TODO:
        - ml.vcontour cannot be added to an ax. Create issue?
        - ml.vcontour does not use model coordinates in plot. Create issue?

    """
    # prepare contour layout
    win = [
        df_model_input.loc["x_channel_max", scen],
        df_model_input.loc["x_hinterland_max", scen],
        0,
        1,
    ]
    # min and max heads within window
    h_min_level = np.floor(
        np.nanmin(ml.headgrid(np.arange(win[0], win[1]), np.arange(win[2], win[3])))
    )
    h_max_level = np.ceil(
        np.nanmax(ml.headgrid(np.arange(win[0], win[1]), np.arange(win[2], win[3])))
    )

    vc_plot = ml.vcontour(
        win,
        n=100,
        labels=True,
        decimals=1,
        levels=np.arange(h_min_level, h_max_level + dh, dh),
        newfig=True,
        figsize=figsize,
    )

    ax = vc_plot.axes
    # secondary x-axis for input dimensions
    ax_input = ax.twiny()
    plot_levee(df_model_input, scen, ax_input)
    plot_bc(ml, ax_input, df_model_input=df_model_input, scen=scen)

    # plot aquifers
    layers_per_aquifer_on_ax(ml, ax)

    # final things
    ax.set_xlim(
        [
            0,
            df_model_input.loc["x_hinterland_max", scen]
            - df_model_input.loc["x_channel_max", scen],
        ]
    )
    ax_input.set_xlim([win[0], win[1]])
    ax_input.set_xlim(
        [
            df_model_input.loc["x_channel_max", scen],
            df_model_input.loc["x_hinterland_max", scen],
        ]
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    # legend, exclude dubbele items
    handles, labels = ax_input.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_input.legend(by_label.values(), by_label.keys(), loc="lower right", ncol=1)

    # assen met titel
    ax.set_title(scen)
    ax.set_xlabel("afstand linkerzijde model (m)")
    ax_input.set_xlabel("afstand referentielijn (m)")
    ax.set_ylabel("(m NAP)")

    return vc_plot


def plot_levee(df_model_input, scen, ax):
    """
    Adds an outline of a levee to plot. For visualisation purposes only.

    Parameters
    ----------
    df_model_input : pd.DataFrame
        Table of model input..
    ax : ax
        ax

    Returns
    -------
    None.

    """
    # calculate plot coordinates
    x_plot = [
        df_model_input.loc["x_outer_toe", scen],
        -0.5 * df_model_input.loc["crest_width", scen],
        0.5 * df_model_input.loc["crest_width", scen],
        df_model_input.loc["x_inner_toe", scen],
    ]

    z_plot = [
        df_model_input.loc["z_groundlevel", scen],
        df_model_input.loc["z_top_levee", scen],
        df_model_input.loc["z_top_levee", scen],
        df_model_input.loc["z_groundlevel", scen],
    ]

    # do plotting
    ax.plot(x_plot, z_plot, color="g", lw=4, label="waterkering (indicatief)")


def plot_bc_elementlist(
    ml,
    ax,
    df_model_input=None,
    scen=None,
    only_h=False,
    plot_type="scatter",
    h_marker="2",
):
    """Plot boundary conditions to ax.

    TODO:
        - created via example model, not fail proof

    Parameters
    ----------
    ml : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    zaq_mid = np.mean([ml.aq.zaqbot, ml.aq.zaqtop], axis=0)

    all_x_inhom_plotted = []

    for element in ml.elementlist:
        # type determines how some variables are called
        if isinstance(element, tml.constant.ConstantStar):
            plot_x = None
            plot_h = None
            marker = None
        elif isinstance(element, tml.linesink1d.HeadLineSink1D):
            plot_x = element.xc
            plot_h = element.hc
            marker = "o"
        elif isinstance(element, tml.linesink1d.HeadDiffLineSink1D):
            plot_x = [element.xc]
            plot_h = None
            marker = "d"
        elif isinstance(element, tml.linesink1d.FluxDiffLineSink1D):
            plot_x = [element.xc]
            plot_h = None
            marker = "d"
        elif isinstance(element, tml.constant.Constant):
            plot_x = [element.xr]
            plot_h = [element.hr]
            marker = "d"
        else:
            print(f"unknown element {element}, data is not plotted")
            plot_x = None
            plot_h = None
            marker = None

        try:
            index_lastpart = element.label.split("_id")[0]
            color = df_model_input.loc[f"color_{index_lastpart}", scen]

            # create name for label
            label_txt = (
                element.label.split("_id")[0]
                .replace("_max", "")
                .replace("_min", "")
                .replace("x_", "")
            )
        except:
            color = "k"
            label_txt = ""

        if plot_h is None and plot_x is not None:
            # plot inhomogenities
            for x in plot_x:
                if x not in all_x_inhom_plotted:
                    ax.axvline(
                        x=x, label="inhomogenity", color="darkgray", lw=0.5, ls="--"
                    )
                    all_x_inhom_plotted.append(x)

        if plot_h is not None:
            # plot headline sinks
            if only_h == False:
                # plot dot in every layer
                p = ax.scatter(
                    [plot_x[0]] * len(element.layers),
                    zaq_mid[element.layers],
                    30,
                    c=color,
                    edgecolors="k",
                    marker=marker,
                    label=f"BC: {label_txt} midden modellaag",
                )

            # plot unique levels
            for h in np.unique(plot_h):
                if plot_type == "scatter":
                    ax.scatter(
                        plot_x[0],
                        h,
                        30,
                        marker=h_marker,  #'_',
                        color=color,
                        label=f"BC: {label_txt}, h={h}",
                    )
                elif plot_type == "vline":
                    ax.axvline(
                        x=plot_x[0],
                        ls="--",
                        lw=0.5,
                        alpha=0.8,
                        color=color,
                        label=f"BC: {label_txt}, h={h}",
                    )
                else:
                    print(f"ongeldige plot_type {plot_type}")


def plot_bc(
    ml,
    ax,
    type_bc="all",
    df_model_input=None,
    scen=None,
    only_h=False,
    plot_type="scatter",
    h_marker="2",
    verbose=False,
):
    """Plot boundary conditions to ax.

    TODO:
        - created via example model, not fail proof

    Parameters
    ----------
    ml : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    zaq_mid = np.mean([ml.aq.zaqbot, ml.aq.zaqtop], axis=0)

    if type_bc in ["all", "elementlist"]:
        # plot elemnts
        for element in ml.elementlist:
            # type determines how some variables are called
            if isinstance(element, tml.linesink1d.HeadLineSink1D):
                plot_x = element.xc
                plot_h = element.hc
                marker = "o"
            elif isinstance(element, tml.constant.Constant):
                plot_x = [element.xr]
                plot_h = [element.hr]
                marker = "d"
            else:
                if verbose:
                    print(f"unknown element {element}, data is not plotted")
                plot_x = None
                plot_h = None

            try:
                index_lastpart = element.label.split("_id")[0]
                color = df_model_input.loc[f"color_{index_lastpart}", scen]

                # create name for label
                label_txt = (
                    element.label.split("_id")[0]
                    .replace("_max", "")
                    .replace("_min", "")
                    .replace("x_", "")
                )
            except:
                color = "k"
                label_txt = ""

            if plot_h is not None:
                # plot headline sinks
                if only_h == False:
                    # plot dot in every layer
                    ax.scatter(
                        [plot_x[0]] * len(element.layers),
                        zaq_mid[element.layers],
                        30,
                        c=color,
                        edgecolors="k",
                        marker=marker,
                        label=f"BC: {label_txt} midden modellaag",
                    )

                # plot unique levels
                for h in np.unique(plot_h):
                    if plot_type == "scatter":
                        ax.scatter(
                            plot_x[0],
                            h,
                            30,
                            marker=h_marker,  #'_',
                            color=color,
                            label=f"BC: {label_txt}, h={h}",
                        )
                    elif plot_type == "vline":
                        ax.axvline(
                            x=plot_x[0],
                            ls="--",
                            lw=0.5,
                            alpha=0.8,
                            color=color,
                            label=f"BC: {label_txt}, h={h}",
                        )
                    else:
                        print(f"ongeldige plot_type {plot_type}")
    if type_bc in ["all", "inhomlist"]:
        # plot hstar from inhom
        colors_inhom = ["darkblue", "royalblue", "deepskyblue", "cadetblue", "seagreen"]
        all_x_max = []
        for i, inhom in enumerate(ml.aq.inhomlist):
            if inhom.x1 == -np.inf:
                x_min = df_model_input.loc["x_channel_max", scen]
            else:
                x_min = inhom.x1
                
            if inhom.x2 == np.inf:
                x_max = df_model_input.loc["x_hinterland_max", scen]
            else:
                x_max = inhom.x2
                
            if inhom.x1 == -np.inf:
                label_x = f"x<{x_max}"
            elif inhom.x2 == np.inf:
                label_x = f"x>{x_min}"
            else:
                label_x = f"{x_min}<x<{x_max}"

            if inhom.hstar is None:
                continue

            if only_h and (plot_type == "vline"):
                ax.axvline(
                    x=x_min,
                    ls="--",
                    lw=0.5,
                    alpha=0.8,
                    color=colors_inhom[i],
                    # only label imhom, not start and end
                    # label=f"BC: start inhom{i}, h={inhom.hstar:0.1f}",
                    label=f"BC: inhom{i}, h={inhom.hstar:0.1f}",
                )
                ax.axvline(
                    x=x_max,
                    ls="--",
                    lw=0.5,
                    alpha=0.8,
                    color=colors_inhom[i],
                    # label=f"BC: end inhom{i}, h={inhom.hstar:0.1f}",
                )

            else:
                ax.plot(
                    [x_min, x_max],
                    [inhom.hstar] * 2,
                    lw=2,
                    color=colors_inhom[i],
                    label=f"BC: inhom{i}, h={inhom.hstar:0.1f}, {label_x}",
                )
            all_x_max.append(x_max)

        # plot hstar model
        x_min = np.asarray(all_x_max).max()


def get_layers_per_aquifer(ml):
    # prepare for sum per aquifer
    first_layer_per_aquifer = []
    last_layer_per_aquifer = []

    last_k = None

    for lyr, k in enumerate(ml.aq.kaq):
        if k != last_k:
            # new lithology
            # add previous layer to list of last layers
            if lyr != 0:
                last_layer_per_aquifer.append(lyr - 1)
            # add current layer to list of new layers
            first_layer_per_aquifer.append(lyr)
            last_k = k
    # add last layer
    last_layer_per_aquifer.append(lyr)

    return first_layer_per_aquifer, last_layer_per_aquifer


def layers_per_aquifer_on_ax(ml, ax):
    first_layer_per_aquifer, last_layer_per_aquifer = get_layers_per_aquifer(ml)

    for first_layer_per_lithology in first_layer_per_aquifer:
        ax.axhline(
            y=ml.aq.zaqtop[first_layer_per_lithology], ls="--", lw=2, color="darkgray"
        )


def plot_q_and_disvec(ml, df_model_input, scen, plot_layers=None, figsize=(12, 10),):
    # prepare discharge plotting locations
    zaq_mid = np.mean([ml.aq.zaqbot, ml.aq.zaqtop], axis=0)

    x_plots_q = [
        np.mean(
            [
                df_model_input.loc["x_outer_toe", scen],
                df_model_input.loc["x_channel_min", scen],
            ]
        ),
        df_model_input.loc["x_outer_toe", scen],
        0,
        # np.mean([df_model_input.loc['x_first_ditch_min', scen], df_model_input.loc['x_first_ditch_max', scen]])]
        df_model_input.loc["x_first_ditch_min", scen] - 0.5,
        np.mean(
            [
                df_model_input.loc["x_first_ditch_max", scen],
                df_model_input.loc["x_hinterland_max", scen],
            ]
        ),
    ]
    labels_q = [
        "halverwege voorland",
        "buitenteen",
        "referentielijn",
        "0.5 m links teensloot",
        "halverwege teensloot en achterland",
    ]

    # prepare for sum per aquifer
    first_layer_per_aquifer, last_layer_per_aquifer = get_layers_per_aquifer(ml)

    # get heads for all layers
    win = [
        df_model_input.loc["x_channel_max", scen],
        df_model_input.loc["x_hinterland_max", scen],
        0,
        1,
    ]

    x = np.linspace(win[0], win[1], 101)
    q = ml.disvecalongline(x, np.zeros_like(x))
    
    # get qz
    df_aq = qz_over_xrange(ml,
                           df_model_input.loc["x_channel_max", scen], 
                           df_model_input.loc["x_hinterland_max", scen],
                           step=1)

    # create figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(scen)

    gs = gridspec.GridSpec(3, len(labels_q))
    ax_qx = fig.add_subplot(gs[0, :])
    ax_qz = fig.add_subplot(gs[1, :])
    axes_q = []
    for i in range(len(labels_q)):
        if i == 0:
            axes_q.append(fig.add_subplot(gs[2, i]))
        else:
            axes_q.append(fig.add_subplot(gs[2, i], sharex=axes_q[0], sharey=axes_q[0]))

    labels_h = []    
    if plot_layers is None:    
        plot_layers = first_layer_per_aquifer
        # append last layer
        plot_layers.append(len(ml.aq.kaq)-1)
    
        
        for i, plot_layer in enumerate(plot_layers):
            if plot_layer == 0:
                labels_h.append("deklaag")
            elif plot_layer > 0:
                wvp_nr = i + 1 - 1
                labels_h.append(f"bovenkant WVP{wvp_nr}")
            else:
                wvp_nr = labels_h[-1].split("WVP")[-1]
                labels_h.append(f"onderkant WVP{wvp_nr}")
    else:
        for plot_layer in plot_layers:
            # simple loop to add empty values when custom plot_layers are selected
            labels_h.append("")
        
        
        

    all_plots = []
    # plot layers
    for label_h, plot_layer in zip(labels_h, plot_layers):
        p = ax_qx.plot(
            x,
            q[0][plot_layer],
            label=f"berekend {label_h} z={zaq_mid[plot_layer]:0.1f}",
        )
        all_plots.append(p)
        
        # plot qz in second plot
        ax_qz.plot(df_aq.loc[plot_layer], 
                   color= p[0].get_color(),
                   )

    plot_bc(
        ml,
        ax_qx,
        df_model_input=df_model_input,
        scen=scen,
        only_h=True,
        plot_type="vline",
    )
    plot_bc(
        ml,
        ax_qz,
        df_model_input=df_model_input,
        scen=scen,
        only_h=True,
        plot_type="vline",
    )

    ax_qx.text(
        -0.01,
        1,
        "naar\n achterland",
        transform=ax_qx.transAxes,
        ha="right",
        va="top",
    )
    ax_qx.text(
        -0.01,
        0,
        "naar\nbuitenwater",
        transform=ax_qx.transAxes,
        ha="right",
        va="bottom",
    )
    ax_qx.set_ylabel("x discharge\n(m3/dag/m1)")
    ax_qx.grid(True)
    
    ax_qz.text(
        -0.01,
        1,
        "omhoog",
        transform=ax_qz.transAxes,
        ha="right",
        va="top",
    )
    ax_qz.text(
        -0.01,
        0,
        "omlaag",
        transform=ax_qz.transAxes,
        ha="right",
        va="bottom",
    )
    
    ax_qz.set_ylabel("z discharge\n(m3/dag/m1)")
    ax_qz.grid(True)


    all_qx = []

    # loop over discharge locations
    for ax, plot_x, label_q in zip(axes_q, x_plots_q, labels_q):
        # loop per plot location
        ax_qx.axvline(x=plot_x, label="locatie Q", ls="--", color="gray")
        ax_qz.axvline(x=plot_x, label="locatie Q", ls="--", color="gray")
        # get data
        qx, qy = ml.disvec(plot_x, 0)
        # plot model results
        ax.plot(
            qx,
            zaq_mid,
            marker="o",
            alpha=0.5,
            markersize=10,
            lw=1,
            color="b",
            label=f"qx (max={qx.max()})",
        )
        ax.set_title(f"{label_q}\nx={plot_x}", fontsize=8)

        # calculate sum per lithology
        for first_layer_this_lithology, last_layer_this_lithology in zip(
            first_layer_per_aquifer, last_layer_per_aquifer
        ):
            qx_this_lithology = qx[
                first_layer_this_lithology : last_layer_this_lithology + 1
            ].sum()
            y_plot_this_lithology = [
                zaq_mid[first_layer_this_lithology],
                zaq_mid[last_layer_this_lithology],
            ]
            ax.plot(
                [qx_this_lithology] * 2,
                y_plot_this_lithology,
                lw=2,
                color="b",
                label="qx per lithology",
            )
            # box with qx
            props = dict(
                boxstyle="round",
                facecolor="lightblue",
            )
            textstr = f"qx\n{qx_this_lithology:0.1f}"
            ax.text(
                qx_this_lithology,
                np.mean(y_plot_this_lithology),
                textstr,
                fontsize=8,
                verticalalignment="center_baseline",
                ha="center",
                bbox=props,
            )
            all_qx.append(qx)
            
        ax.axhline(y=ml.aq.zaqtop[0], lw=1.5, color="g")
        for bot in ml.aq.zaqbot:
            ax.axhline(y=bot, lw=0.5, color="gray")

        for p, plot_layer in zip(all_plots, plot_layers):
            ax.axhline(y=zaq_mid[plot_layer], ls="--", lw=1, color=p[0].get_color())

        layers_per_aquifer_on_ax(ml, ax)

        ax.set_xlabel("discharge (m3/dag/m1)")

    axes_q[0].set_ylabel("m NAP")

    handles, labels = ax_qx.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_qx.legend(by_label.values(), by_label.keys(), loc="upper right", ncol=1)

    for ax in [ax_qx, ax_qz]:
        ax.set_xlim(
            [
                df_model_input.loc["x_channel_max", scen],
                df_model_input.loc["x_hinterland_max", scen],
            ]
        )
    
    return fig, ax, all_qx

def qz_over_xrange(ml, xmin, xmax, step=1):
    """
    Calculate discharge between layers (qz) over an x range, uses qz_at_x

    Parameters
    ----------
    ml : timml.Model
        TimML groundwater model.
    xmin : float
        minimum value of x range.
    xmax : float
        maximum value of x range.
    step : float, optional
        step size of range. The default is 1.

    Returns
    -------
    df : pd.DataFrame
        qz per x location over all layers.

    """
    all_x = []
    all_qz = []
    for x in np.arange(xmin, xmax, step):
        all_x.append(x)
        all_qz.append(qz_at_x(ml, x))
        
    #print(qz_at_x(ml, x), all_qz)
    df = pd.DataFrame(np.asarray(all_qz).T, columns=all_x)   
    
    return df

def qz_at_x(ml, x, y=0):
    """
    Calculate discharge between layers (qz) at a specific x location.
    With thanks to Davìd Brakenhoff

    Parameters
    ----------
    ml : timml.Model
        TimML groundwater model.
    x : float
        x coordinate.
    y : float, optional
        y coordinate. The default is 0.

    Returns
    -------
    qzlayer : list
        qz between all layers.

    """
    
    aq = ml.aq.find_aquifer_data(x, y)
    h = ml.head(x, y, aq=aq)
    qzlayer = np.zeros(aq.naq + 1)
    qzlayer[1:-1] = (h[1:] - h[:-1]) / aq.c[1:]
    if aq.ltype[0] == 'l':
        qzlayer[0] = (h[0] - aq.hstar) / aq.c[0]
    
    return qzlayer



def plot_contour_and_tracelines(
    ml,
    df_model_input,
    scen,
    xstart,
    ystart,
    zstart,
    hstepmax,
    dh=1,
):
    win = [
        df_model_input.loc["x_channel_max", scen],
        df_model_input.loc["x_hinterland_max", scen],
        0,
        1,
    ]

    h_min_level = np.floor(
        np.nanmin(ml.headgrid(np.arange(win[0], win[1]), np.arange(win[2], win[3])))
    )
    h_max_level = np.ceil(
        np.nanmax(ml.headgrid(np.arange(win[0], win[1]), np.arange(win[2], win[3])))
    )

    ml.vcontour(
        win,
        n=100,
        labels=True,
        decimals=1,
        levels=np.arange(h_min_level, h_max_level + dh, dh),
    )

    ml.tracelines(
        xstart=xstart, ystart=ystart, zstart=zstart, hstepmax=hstepmax, win=win
    )


def plot_h(
    ml, df_model_input, scen, figsize=(12, 4), plot_layers=None, do_plot_bc=True
):
    # prepare discharge plotting locations
    zaq_mid = np.mean([ml.aq.zaqbot, ml.aq.zaqtop], axis=0)

    # prepare for sum per aquifer
    first_layer_per_aquifer, last_layer_per_aquifer = get_layers_per_aquifer(ml)

    # get heads for all layers
    win = [
        df_model_input.loc["x_channel_max", scen],
        df_model_input.loc["x_hinterland_max", scen],
        0,
        1,
    ]

    x = np.linspace(win[0], win[1], 101)
    h = ml.headalongline(x, np.zeros_like(x))

    # create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(scen)

    if plot_layers is None:
        plot_layers = first_layer_per_aquifer
        plot_layers.append(-1)

    labels = []
    for i, plot_layer in enumerate(plot_layers):
        if plot_layer == 0:
            labels.append("deklaag")
        elif plot_layer > 0:
            wvp_nr = i + 1 - 1
            labels.append(f"bovenkant WVP{wvp_nr}")
        else:
            wvp_nr = labels[-1].split("WVP")[-1]
            labels.append(f"onderkant WVP{wvp_nr}")

    # plot layers
    for label, plot_layer in zip(labels, plot_layers):
        ax.plot(
            x,
            h[plot_layer],
            label=f"berekend {label} z={zaq_mid[plot_layer]:0.1f}",
        )
    if do_plot_bc:
        plot_bc(
            ml,
            ax,
            df_model_input=df_model_input,
            scen=scen,
            only_h=True,
        )

    ax.set_xlabel("afstand (m ref)")
    ax.set_ylabel("(m NAP)")

    ax.set_xlim([win[0], win[1]])

    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", ncol=1)

    return fig, ax


def calculate_q_semi(ml, verbose=False):
    last_hls_name = ""
    discharge_per_bc = []
    name_per_bc = []
    dicharge_this_boundary = []

    for i, element in enumerate(ml.elementlist):
        # print(f'element {i}')

        if isinstance(element, tml.linesink1d.HeadLineSink1D):
            if verbose:
                print(f"add {element}")

            # nieuwe boundary?
            this_hls_name = element.label.split("_id")[0]
            if this_hls_name != last_hls_name:
                # yes, calculate sum previous and store
                # calculate total and store to list
                discharge_per_bc.append(np.asarray(dicharge_this_boundary).sum())
                name_per_bc.append(last_hls_name)

                # reset list
                dicharge_this_boundary = []

            discharge_this_bc = element.discharge()
            dicharge_this_boundary.append(discharge_this_bc)

            last_hls_name = this_hls_name
        else:
            if verbose:
                print(f"skip {element}")

    # save last one
    discharge_per_bc.append(np.asarray(dicharge_this_boundary).sum())
    name_per_bc.append(last_hls_name)

    if verbose:
        for name, discharge in zip(name_per_bc, discharge_per_bc):
            print(name, discharge)

    # return from id=1, id=0 is empty list value
    return name_per_bc[1:], discharge_per_bc[1:]


def sensitity_one_parameter(
    df_model_input,
    base_scen,
    parameter,
    factors_multiply=None,
    factors_add=None,
    nr_digits_df=1,
    close_plot=True,
    dh=0.5,
    verbose=False,
):
    """
    Do a sensitivity analysis on one parameter

    Parameters
    ----------
    df_model_input : pd.DataFrame
        Model input.
    base_scen : str
        Colname where scenario is on based.
    parameter : str
        Parameter to change in analysis
    factors_multiply : float or list, optional
        Factor to multiply base scen with. The default is None.
    factors_add : float or list, optional
        Factors to add to base scen, after muliply. The default is None.
    nr_digits_df : int, optional
        Round discharge results to this number. The default is 1.
    close_plot : bool, optional
        Close all plots. The default is True.
    dh : float, optional
        Interval for contour plot. The default is 0.5.
    verbose : bool, optional
        Print some info during rung. The default is False

    Returns
    -------
    pd.DataFrame
        Discharge results.

    """
    # assign factor 1 when None
    if factors_multiply is None:
        factors_multiply = [1]
    if factors_add is None:
        factors_add = [0]

    # change input to list
    if not isinstance(factors_multiply, list):
        factors_multiply = [factors_multiply]
    if not isinstance(factors_add, list):
        factors_add = [factors_add]

    dict_q = defaultdict(list)
    dict_q["name"] = []
    dict_q["factor_multiply"] = []
    dict_q["factor_add"] = []
    #dict_q["q_bc_hydr_load"] = []
    #dict_q["q_bc_foreland"] = []
    dict_q["q_bc_ditch"] = []
    dict_q["q_bc_hinterland"] = []

    dict_q["q_refline_dek"] = []
    dict_q["q_refline_WVP1"] = []
    dict_q["q_refline_WVP2"] = []
    dict_q["q_refline_WVP_perc"] = []
    dict_q["q_refline_array"] = []

    for i, factor_multiply in enumerate(factors_multiply):
        for j, factor_add in enumerate(factors_add):
            # scenario name
            this_scen = f"{base_scen}_run_{parameter}"
            if factor_multiply is not None:
                # parameter_multiply given, add value
                this_scen += f"_m{factor_multiply}"
            if factor_add is not None:
                # add add part to name
                this_scen += f"_a{factor_add}"

            # copy input values
            df_model_input[this_scen] = df_model_input[base_scen]

            # what is type of parameter?
            if isinstance(df_model_input.loc[parameter, this_scen], list):
                # list requires special operation

                # update values in list
                in_df = df_model_input.loc[parameter, this_scen]
                new_lst = []

                for value_in_lst in in_df:
                    new_lst.append([(value_in_lst[0] * factor_multiply) + factor_add])
                # to dataframe
                df_model_input.at[parameter, this_scen] = new_lst
            else:
                # df_model_input.at[parameter,this_scen].multiply(factor_multiply).plus(factor_add)
                df_model_input.at[parameter, this_scen] = (
                    df_model_input.at[parameter, this_scen] * factor_multiply
                ) + factor_add

            # run model
            (
                ml,
                #df_ml_layer,
                q_names,
                q_values,
                first_layer_per_aquifer,
                last_layer_per_aquifer,
            ) = create_plot_model(
                df_model_input, this_scen, dh=dh, close_plot=close_plot, verbose=verbose
            )

            # add results to dictonary
            dict_q["name"].append(f"{this_scen}")
            dict_q["factor_multiply"].append(factor_multiply)
            dict_q["factor_add"].append(factor_add)
            #dict_q["q_bc_hydr_load"].append(np.round(q_values[0], nr_digits_df))
            #dict_q["q_bc_foreland"].append(np.round(q_values[1], nr_digits_df))
            dict_q["q_bc_ditch"].append(np.round(q_values[0], nr_digits_df))
            dict_q["q_bc_hinterland"].append(np.round(q_values[1], nr_digits_df))

            """
            q_ref = q_values[5]
            dict_q["q_refline_dek"].append(np.round(q_ref[first_layer_per_aquifer[0]:last_layer_per_aquifer[0]].sum(), nr_digits_df))
            q_refline_WVP1 = q_ref[first_layer_per_aquifer[1]:last_layer_per_aquifer[1]].sum()
            dict_q["q_refline_WVP1"].append(np.round(q_refline_WVP1, nr_digits_df))
            q_refline_WVP2 = q_ref[first_layer_per_aquifer[2]:last_layer_per_aquifer[2]].sum()
            dict_q["q_refline_WVP2"].append(np.round(q_refline_WVP2, nr_digits_df))
            dict_q["q_refline_WVP_perc"].append(np.round((q_refline_WVP1 / q_refline_WVP2) * 100, 1))
            dict_q["q_refline_array"].append(np.round(q_ref, nr_digits_df))
            """
            (
                q_refline_dek,
                q_refline_WVP1,
                q_refline_WVP2,
                q_refline_WVP_perc,
                q_ref,
            ) = calc_q_per_aquifer(
                q_values,
                first_layer_per_aquifer,
                last_layer_per_aquifer,
                q_analyse_id=2,
            )
            dict_q["q_refline_dek"].append(np.round(q_refline_dek, nr_digits_df))
            dict_q["q_refline_WVP1"].append(np.round(q_refline_WVP1, nr_digits_df))
            dict_q["q_refline_WVP2"].append(np.round(q_refline_WVP2, nr_digits_df))
            dict_q["q_refline_WVP_perc"].append(
                np.round(q_refline_WVP_perc, nr_digits_df)
            )
            dict_q["q_refline_array"].append(q_ref)

    return pd.DataFrame(dict_q)


def sensitity_one_parameter_semi(
    df_model_input,
    base_scen,
    parameter,
    factors_multiply=None,
    factors_add=None,
    nr_digits_df=1,
    close_plot=True,
    dh=0.5,
    verbose=False,
):
    """
    Do a sensitivity analysis on one parameter

    Parameters
    ----------
    df_model_input : pd.DataFrame
        Model input.
    base_scen : str
        Colname where scenario is on based.
    parameter : str
        Parameter to change in analysis
    factors_multiply : float or list, optional
        Factor to multiply base scen with. The default is None.
    factors_add : float or list, optional
        Factors to add to base scen, after muliply. The default is None.
    nr_digits_df : int, optional
        Round discharge results to this number. The default is 1.
    close_plot : bool, optional
        Close all plots. The default is True.
    dh : float, optional
        Interval for contour plot. The default is 0.5.
    verbose : bool, optional
        Print some info during rung. The default is False

    Returns
    -------
    pd.DataFrame
        Discharge results.

    """
    # assign factor 1 when None
    if factors_multiply is None:
        factors_multiply = [1]
    if factors_add is None:
        factors_add = [0]

    # change input to list
    if not isinstance(factors_multiply, list):
        factors_multiply = [factors_multiply]
    if not isinstance(factors_add, list):
        factors_add = [factors_add]

    dict_q = defaultdict(list)
    dict_q["name"] = []
    dict_q["factor_multiply"] = []
    dict_q["factor_add"] = []
    dict_q["q_bc_ditch"] = []

    dict_q["q_refline_dek"] = []
    dict_q["q_refline_WVP1"] = []
    dict_q["q_refline_WVP2"] = []
    dict_q["q_refline_WVP_perc"] = []
    dict_q["q_refline_array"] = []

    for i, factor_multiply in enumerate(factors_multiply):
        for j, factor_add in enumerate(factors_add):
            # scenario name
            this_scen = f"{base_scen}_run_{parameter}"
            if factor_multiply is not None:
                # parameter_multiply given, add value
                this_scen += f"_m{factor_multiply}"
            if factor_add is not None:
                # add add part to name
                this_scen += f"_a{factor_add}"

            # copy input values
            df_model_input[this_scen] = df_model_input[base_scen]

            # what is type of parameter?
            if isinstance(df_model_input.loc[parameter, this_scen], list):
                # list requires special operation

                # update values in list
                in_df = df_model_input.loc[parameter, this_scen]
                new_lst = []

                for value_in_lst in in_df:
                    new_lst.append([(value_in_lst * factor_multiply) + factor_add])
                # to dataframe
                df_model_input.at[parameter, this_scen] = new_lst
            else:
                # df_model_input.at[parameter,this_scen].multiply(factor_multiply).plus(factor_add)
                df_model_input.at[parameter, this_scen] = (
                    df_model_input.at[parameter, this_scen] * factor_multiply
                ) + factor_add

            # run model
            (
                ml,
                df_ml_layer,
                q_names,
                q_values,
                first_layer_per_aquifer,
                last_layer_per_aquifer,
            ) = create_plot_model(
                df_model_input, this_scen, dh=dh, close_plot=close_plot, verbose=verbose
            )

            # add results to dictonary
            dict_q["name"].append(f"{this_scen}")
            dict_q["factor_multiply"].append(factor_multiply)
            dict_q["factor_add"].append(factor_add)
            dict_q["q_bc_ditch"].append(np.round(q_values[0], nr_digits_df))

            (
                q_refline_dek,
                q_refline_WVP1,
                q_refline_WVP2,
                q_refline_WVP_perc,
                q_ref,
            ) = calc_q_per_aquifer(
                q_values,
                first_layer_per_aquifer,
                last_layer_per_aquifer,
                q_analyse_id=-1,
            )
            dict_q["q_refline_dek"].append(np.round(q_refline_dek, nr_digits_df))
            dict_q["q_refline_WVP1"].append(np.round(q_refline_WVP1, nr_digits_df))
            dict_q["q_refline_WVP2"].append(np.round(q_refline_WVP2, nr_digits_df))
            dict_q["q_refline_WVP_perc"].append(
                np.round(q_refline_WVP_perc, nr_digits_df)
            )
            dict_q["q_refline_array"].append(q_ref)

    return pd.DataFrame(dict_q)


def calc_q_per_aquifer(
    q_values,
    first_layer_per_aquifer,
    last_layer_per_aquifer,
    q_analyse_id=-1,
    do_print=False,
):
    q_ref = q_values[q_analyse_id]
    q_refline_dek = q_ref[
        first_layer_per_aquifer[0] : last_layer_per_aquifer[0]
    ].sum()
    q_refline_WVP1 = q_ref[
        first_layer_per_aquifer[1] : last_layer_per_aquifer[1]
    ].sum()
    q_refline_WVP2 = q_ref[
        first_layer_per_aquifer[2] : last_layer_per_aquifer[2]
    ].sum()
    q_refline_WVP_perc = (q_refline_WVP1 / q_refline_WVP2) * 100

    if do_print:
        print(
            f"debiet op referentielijn (x=0): WVP1={q_refline_WVP1:0.1f}, WVP2={q_refline_WVP2:0.1f}, percentage={q_refline_WVP_perc:0.1f}"
        )

    return q_refline_dek, q_refline_WVP1, q_refline_WVP2, q_refline_WVP_perc, q_ref

