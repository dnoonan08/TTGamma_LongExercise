import matplotlib.pyplot as plt
from cycler import cycler
from coffea import hist
import numpy as np

def plotWithRatio(h, hData, overlay, stacked=True, density=False, invertStack=True, lumi=35.9, label="CMS Preliminary",colors=None,ratioRange=[0.5,1.5], xRange=None, yRange=None, logY=False,extraText = None, leg='upper right', binwnorm=None):

    # make a nice ratio plot
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    if not hData is None:
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
        fig.subplots_adjust(hspace=.07)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    # Here is an example of setting up a color cycler to color the various fill patches
    # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=6
    from cycler import cycler
    if not colors is None:
        if invertStack: 
            _n = len(h.identifiers(overlay))-1
            colors = colors[_n::-1]
        ax.set_prop_cycle(cycler(color=colors))

    fill_opts = {
        'edgecolor': (0,0,0,0.3),
        'alpha': 0.8
    }
    error_opts = {
        'label':'Stat. Unc.',
        'hatch':'///',
        'facecolor':'none',
        'edgecolor':(0,0,0,.5),
        'linewidth': 0
    }
    if not stacked:
        error_opts = None
        fill_opts = None
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
#        'emarker': '_'
    }

    if invertStack:
        if type(h._axes[0])==hist.hist_tools.Cat:
            h._axes[0]._sorted.reverse()
    hist.plot1d(h,
                overlay=overlay,
                ax=ax,
                clear=False,
                stack=stacked,
                density=density,
                line_opts=None,
                fill_opts=fill_opts,
                error_opts=error_opts,
                binwnorm=binwnorm
               )
    # if invertStack:
    #     if type(h._axes[0])==hist.hist_tools.Cat:
    #         h._axes[0]._sorted.reverse()

    if not hData is None:
        hist.plot1d(hData,
#                    overlay=overlay,
                    ax=ax,
                    clear=False,
                    error_opts=data_err_opts,
                    binwnorm=binwnorm
                   )

    ax.autoscale(axis='x', tight=True)
    ax.set_ylim(0, None)
    if not binwnorm is None:
        ax.set_ylabel(f"<Counts/{binwnorm}>")
        if '[' in ax.get_xlabel():
            units = ax.get_xlabel().split('[')[-1].split(']')[0]
            ax.set_ylabel(f"<Counts / {binwnorm} {units}>")
            
    ax.set_xlabel(None)
    
    if leg=="right":
        leg_anchor=(1., 1.)
        leg_loc='upper left'
    elif leg=="upper right":
        leg_anchor=(1., 1.)
        leg_loc='upper right'
    elif leg=="upper left":
        leg_anchor=(0., 1.)
        leg_loc='upper left'

    if not leg is None:
        handles, labels = ax.get_legend_handles_labels()
        if not hData is None:
            handles = handles[-1:]+handles[:-1]
            labels = ['Data']+labels[:-1]            
        ax.legend(handles, labels,bbox_to_anchor=leg_anchor,loc=leg_loc)


    
    if not hData is None:
        hist.plotratio(hData, h.sum(overlay),
                       ax=rax,
                       error_opts=data_err_opts,
                       denom_fill_opts={},
                       guide_opts={},
                       unc='num'
                      )
        rax.set_ylabel('Ratio')
            
        rax.set_ylim(ratioRange[0],ratioRange[1])

    
    if logY:
        ax.set_yscale("log")
        ax.set_ylim(1,ax.get_ylim()[1]*5)        

    if not xRange is None:
        ax.set_xlim(xRange[0],xRange[1])
    if not yRange is None:
        ax.set_ylim(yRange[0],yRange[1])

    CMS = plt.text(0., 1., r"$\bf{CMS}$ Preliminary",
                    fontsize=16,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax.transAxes
                   )

    if not extraText is None:
        
        extraLabel = plt.text(0.02, .99, extraText,
                        fontsize=16,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes
                       )
        ax.set_ylim(0,ax.get_ylim()[1]*1.1)
    
    lumi = plt.text(1., 1., r"%.1f fb$^{-1}$ (13 TeV)"%(lumi),
                    fontsize=16,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=ax.transAxes
                   )


def SetRangeHist(histogram, axisName, lower_bound=None, upper_bound=None):
    old_axis = histogram.axis(axisName)
    lower_idx, upper_idx = None, None
    if not lower_bound is None:
        lower_idx = np.where(old_axis.edges()>=lower_bound)[0][0]
    if not upper_bound is None:
        upper_idx = np.where(old_axis.edges()<=upper_bound)[0][-1]+1

    new_axis = hist.Bin(old_axis.name, old_axis.label, old_axis.edges()[lower_idx:upper_idx])
    return histogram.rebin(axisName, new_axis)
    
def RebinHist(histogram, axisName, rebinN=1): 
    old_axis = histogram.axis(axisName)
    new_axis = hist.Bin(old_axis.name, old_axis.label, old_axis.edges()[::rebinN])
    return histogram.rebin(axisName, new_axis)
    
        
