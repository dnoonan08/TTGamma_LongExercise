import uproot
from coffea.lookup_tools import dense_lookup


eleIDfiles = {2016:'ScaleFactors/MuEGammaScaleFactors/ele2016/2016LegacyReReco_ElectronTight_Fall17V2.root',
              2017:'ScaleFactors/MuEGammaScaleFactors/ele2017/2017_ElectronTight.root',
              2018:'ScaleFactors/MuEGammaScaleFactors/ele2018/2018_ElectronTight.root',             
             }

eleRecofiles = {2016:'ScaleFactors/MuEGammaScaleFactors/ele2016/egammaEffi.txt_EGM2D_runBCDEF_passingRECO.root',
                2017:'ScaleFactors/MuEGammaScaleFactors/ele2017/egammaEffi.txt_EGM2D_runBCDEF_passingRECO.root',
                2018:'ScaleFactors/MuEGammaScaleFactors/ele2018/egammaEffi.txt_EGM2D_updatedAll.root',             
               }


muSFFiles = {2016: [{'id'   : ("ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
                     'iso'   : ("ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                     'trig'  : ("ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 19.656062760/35.882515396},
                    {'id'     : ("ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
                     'iso'   : ("ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                     'trig'  : ("ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 16.226452636/35.882515396}],
             2017: [{'id'     : ("ScaleFactors/MuEGammaScaleFactors/mu2017/RunBCDEF_SF_ID.root", "NUM_TightID_DEN_genTracks_pt_abseta"),
                     'iso'   : ("ScaleFactors/MuEGammaScaleFactors/mu2017/RunBCDEF_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"),
                     'trig'  : ("ScaleFactors/MuEGammaScaleFactors/mu2017/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root", "IsoMu27_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 1.}],
             2018: [{'id'     : ("ScaleFactors/MuEGammaScaleFactors/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ID.root", "NUM_TightID_DEN_TrackerMuons_pt_abseta"),
                     'iso'   : ("ScaleFactors/MuEGammaScaleFactors/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"),
                     'trig'  : ("ScaleFactors/MuEGammaScaleFactors/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_BeforeMuonHLTUpdate.root", "IsoMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 8.950818835/59.688059536},
                    {'id'     : ("ScaleFactors/MuEGammaScaleFactors/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ID.root", "NUM_TightID_DEN_TrackerMuons_pt_abseta"),
                     'iso'   : ("ScaleFactors/MuEGammaScaleFactors/mu2018/EfficienciesStudies_2018_rootfiles_RunABCD_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta"),
                     'trig'  : ("ScaleFactors/MuEGammaScaleFactors/mu2018/EfficienciesStudies_2018_trigger_EfficienciesAndSF_2018Data_AfterMuonHLTUpdate.root", "IsoMu24_PtEtaBins/abseta_pt_ratio"),
                     'scale' : 50.737240701/59.688059536}],
            }



def getEleSF_lookups(year):

    ele_id_file = uproot.open(eleIDfiles[year])
    ele_id_sf = dense_lookup.dense_lookup(ele_id_file["EGamma_SF2D"].values, ele_id_file["EGamma_SF2D"].edges)
    ele_id_err = dense_lookup.dense_lookup(ele_id_file["EGamma_SF2D"].variances**0.5, ele_id_file["EGamma_SF2D"].edges)

    ele_reco_file = uproot.open(eleRecofiles[year])
    ele_reco_sf = dense_lookup.dense_lookup(ele_reco_file["EGamma_SF2D"].values, ele_reco_file["EGamma_SF2D"].edges)
    ele_reco_err = dense_lookup.dense_lookup(ele_reco_file["EGamma_SF2D"].variances**.5, ele_reco_file["EGamma_SF2D"].edges)

    return ele_id_sf, ele_id_err, ele_reco_sf, ele_reco_err

def getEleSF(pt, eta, year, split=False):

    ele_id_sf, ele_id_err, ele_reco_sf, ele_reco_err = getEleSF_lookups(year)

    eleID = ele_id_sf(eta, pt)
    eleIDerr = ele_id_err(eta, pt)

    eleRECO = ele_reco_sf(eta, pt)
    eleRECOerr = ele_reco_err(eta,pt)
    
    totalSF = eleID*eleRECO
    if not split:
        totalSFup = (eleID + eleIDerr) * (eleRECO + eleRECOerr)
        totalSFdown = (eleID - eleIDerr) * (eleRECO - eleRECOerr)
        return totalSF.prod(), totalSFup.prod(), totalSFdown.prod()
    else:
        totalSF_ID_up = (eleID + eleIDerr) * (eleRECO)
        totalSF_ID_down = (eleID - eleIDerr) * (eleRECO)
        totalSF_RECO_up = (eleID) * (eleRECO + eleRECOerr)
        totalSF_RECO_down = (eleID) * (eleRECO - eleRECOerr)
        return totalSF.prod(), totalSF_ID_up.prod(), totalSF_ID_down.prod(), totalSF_RECO_up.prod(), totalSF_RECO_down.prod()





def getMuSF_lookups(year):
    muSFFileList = muSFFiles[year]
    
    scaleFactors = muSFFileList[0]
    id_file = uproot.open(scaleFactors['id'][0])
    iso_file = uproot.open(scaleFactors['iso'][0])
    trig_file = uproot.open(scaleFactors['trig'][0])

    id_vals = id_file[scaleFactors['id'][1]].values
    id_err = id_file[scaleFactors['id'][1]].variances**0.5
    id_edges = id_file[scaleFactors['id'][1]].edges

    iso_vals = iso_file[scaleFactors['iso'][1]].values
    iso_err = iso_file[scaleFactors['iso'][1]].variances**0.5
    iso_edges = iso_file[scaleFactors['iso'][1]].edges

    trig_vals = trig_file[scaleFactors['trig'][1]].values
    trig_err = trig_file[scaleFactors['trig'][1]].variances**0.5
    trig_edges = trig_file[scaleFactors['trig'][1]].edges

    id_vals *= scaleFactors['scale']
    id_err *= scaleFactors['scale']
    iso_vals *= scaleFactors['scale']
    iso_err *= scaleFactors['scale']
    trig_vals *= scaleFactors['scale']
    trig_err *= scaleFactors['scale']

    for scaleFactors in muSFFileList[1:]:
        id_file = uproot.open(scaleFactors['id'][0])
        iso_file = uproot.open(scaleFactors['iso'][0])
        trig_file = uproot.open(scaleFactors['trig'][0])

        id_vals += id_file[scaleFactors['id'][1]].values * scaleFactors['scale']
        id_err += id_file[scaleFactors['id'][1]].variances**0.5 * scaleFactors['scale']

        iso_vals += iso_file[scaleFactors['iso'][1]].values * scaleFactors['scale']
        iso_err += iso_file[scaleFactors['iso'][1]].variances**0.5 * scaleFactors['scale']

        trig_vals += trig_file[scaleFactors['trig'][1]].values * scaleFactors['scale']
        trig_err += trig_file[scaleFactors['trig'][1]].variances**0.5 * scaleFactors['scale']


        
    id_sf = dense_lookup.dense_lookup(id_vals, id_edges)
    id_err = dense_lookup.dense_lookup(id_err, id_edges)

    iso_sf = dense_lookup.dense_lookup(iso_vals, iso_edges)
    iso_err = dense_lookup.dense_lookup(iso_err, iso_edges)

    trig_sf = dense_lookup.dense_lookup(trig_vals, trig_edges)
    trig_err = dense_lookup.dense_lookup(trig_err, trig_edges)
    
    return id_sf, id_err, iso_sf, iso_err, trig_sf, trig_err

def getMuSF(pt, eta, year, split=False):
    id_sf, id_err, iso_sf, iso_err, trig_sf, trig_err = getMuSF_lookups(year)
    
    if year==2016:
        muID = id_sf(eta,pt)
        muIDerr = id_err(eta,pt)
        
        muIso = iso_sf(eta,pt)
        muIsoerr = iso_err(eta,pt)

        muTrig = iso_sf(abs(eta),pt)
        muTrigerr = iso_err(abs(eta),pt)
    
    if year==2017:
        muID = id_sf(pt,abs(eta))
        muIDerr = id_err(pt,abs(eta))
        
        muIso = iso_sf(pt,abs(eta))
        muIsoerr = iso_err(pt,abs(eta))

        muTrig = iso_sf(abs(eta),pt)
        muTrigerr = iso_err(abs(eta),pt)
    
    if year==2018:
        muID = id_sf(pt,abs(eta))
        muIDerr = id_err(pt,abs(eta))
        
        muIso = iso_sf(pt,abs(eta))
        muIsoerr = iso_err(pt,abs(eta))

        muTrig = iso_sf(abs(eta),pt)
        muTrigerr = iso_err(abs(eta),pt)
    
    totalSF = muID*muIso*muTrig
    if not split:
        totalSF_up = (muID + muIDerr) * (muIso + muIsoerr) * (muTrig + muTrigerr)
        totalSF_down = (muID - muIDerr) * (muIso - muIsoerr) * (muTrig - muTrigerr)
        return totalSF.prod(), totalSF_up.prod(), totalSF_down.prod()
    else:
        totalSF_ID_up = (muID + muIDerr) * (muIso) * (muTrig)
        totalSF_ID_down = (muID - muIDerr) * (muIso) * (muTrig)        
        totalSF_Iso_up = (muID) * (muIso + muIsoerr) * (muTrig)
        totalSF_Iso_down = (muID) * (muIso - muIsoerr) * (muTrig)        
        totalSF_Trig_up = (muID) * (muIso) * (muTrig + muTrigerr)
        totalSF_Trig_down = (muID) * (muIso) * (muTrig - muTrigerr)        
        return totalSF.prod(), totalSF_ID_up.prod(), totalSF_ID_down.prod(), totalSF_Iso_up.prod(), totalSF_Iso_down.prod(), totalSF_Trig_up.prod(), totalSF_Trig_down.prod()
