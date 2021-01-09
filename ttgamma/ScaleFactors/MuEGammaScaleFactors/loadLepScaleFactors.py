from coffea.lookup_tools import dense_lookup
from coffea import util
import uproot

#files for mu scale factors                                                                                                                                  
muSFFileList = [{'id'   : (f"mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ID.root", "NUM_Tigh\
tID_DEN_genTracks_eta_pt"),
                 'iso'   : (f"mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ISO.root", "NUM_Ti\
ghtRelIso_DEN_TightIDandIPCut_eta_pt"),
                 'trig'  : (f"mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root", "IsoMu24_OR_I\
soTkMu24_PtEtaBins/abseta_pt_ratio"),
                 'scale' : 19.656062760/35.882515396},
                {'id'     : (f"mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ID.root", "NUM_Tight\
ID_DEN_genTracks_eta_pt"),
                 'iso'   : (f"mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ISO.root", "NUM_Tight\
RelIso_DEN_TightIDandIPCut_eta_pt"),
                 'trig'  : (f"mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root", "IsoMu24_OR_I\
soTkMu24_PtEtaBins/abseta_pt_ratio"),
                 'scale' : 16.226452636/35.882515396}]


ele_id_file = uproot.open(f'ele2016/2016LegacyReReco_ElectronTight_Fall17V2.root')
ele_id_sf = dense_lookup.dense_lookup(ele_id_file["EGamma_SF2D"].values, ele_id_file["EGamma_SF2D"].edges)
ele_id_err = dense_lookup.dense_lookup(ele_id_file["EGamma_SF2D"].variances**0.5, ele_id_file["EGamma_SF2D"].edges)

ele_reco_file = uproot.open(f'ele2016/egammaEffi.txt_EGM2D_runBCDEF_passingRECO.root')
ele_reco_sf = dense_lookup.dense_lookup(ele_reco_file["EGamma_SF2D"].values, ele_reco_file["EGamma_SF2D"].edges)
ele_reco_err = dense_lookup.dense_lookup(ele_reco_file["EGamma_SF2D"].variances**.5, ele_reco_file["EGamma_SF2D"].edges)


mu_id_vals = 0
mu_id_err = 0
mu_iso_vals = 0
mu_iso_err = 0
mu_trig_vals = 0
mu_trig_err = 0

for scaleFactors in muSFFileList:
    id_file = uproot.open(scaleFactors['id'][0])
    iso_file = uproot.open(scaleFactors['iso'][0])
    trig_file = uproot.open(scaleFactors['trig'][0])
    
    mu_id_vals += id_file[scaleFactors['id'][1]].values * scaleFactors['scale']
    mu_id_err += id_file[scaleFactors['id'][1]].variances**0.5 * scaleFactors['scale']
    mu_id_edges = id_file[scaleFactors['id'][1]].edges
    
    mu_iso_vals += iso_file[scaleFactors['iso'][1]].values * scaleFactors['scale']
    mu_iso_err += iso_file[scaleFactors['iso'][1]].variances**0.5 * scaleFactors['scale']
    mu_iso_edges = iso_file[scaleFactors['iso'][1]].edges
    
    mu_trig_vals += trig_file[scaleFactors['trig'][1]].values * scaleFactors['scale']
    mu_trig_err += trig_file[scaleFactors['trig'][1]].variances**0.5 * scaleFactors['scale']
    mu_trig_edges = trig_file[scaleFactors['trig'][1]].edges

mu_id_sf = dense_lookup.dense_lookup(mu_id_vals, mu_id_edges)
mu_id_err = dense_lookup.dense_lookup(mu_id_err, mu_id_edges)
mu_iso_sf = dense_lookup.dense_lookup(mu_iso_vals, mu_iso_edges)
mu_iso_err = dense_lookup.dense_lookup(mu_iso_err, mu_iso_edges)
mu_trig_sf = dense_lookup.dense_lookup(mu_trig_vals, mu_trig_edges)
mu_trig_err = dense_lookup.dense_lookup(mu_trig_err, mu_trig_edges)

util.save(ele_id_sf, 'ele_id_sf.coffea')
util.save(ele_id_err, 'ele_id_err.coffea')
util.save(ele_reco_sf, 'ele_reco_sf.coffea')
util.save(ele_reco_err, 'ele_reco_err.coffea')


util.save(mu_id_sf, 'mu_id_sf.coffea')
util.save(mu_id_err, 'mu_id_err.coffea')
util.save(mu_iso_sf, 'mu_iso_sf.coffea')
util.save(mu_iso_err, 'mu_iso_err.coffea')
util.save(mu_trig_sf, 'mu_trig_sf.coffea')
util.save(mu_trig_err, 'mu_trig_err.coffea')
