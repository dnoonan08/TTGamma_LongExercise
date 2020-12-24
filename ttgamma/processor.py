import time

from coffea import hist, util
#from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty, JetTransformer, JetResolution, JetResolutionScaleFactor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from functools import partial
import uproot
import awkward1 as ak
#from awkward import JaggedArray
import numpy as np
import pickle
import sys
from coffea.lookup_tools import extractor, dense_lookup
import numba
import re

from .utils.plotting import plotWithRatio
from .utils.crossSections import *
from .utils.efficiencies import getMuSF, getEleSF

from packaging import version
import coffea
if (version.parse(coffea.__version__) < version.parse('0.6.47') ):
    raise Exception('Code requires coffea version 0.6.47 or newer')

import os.path
cwd = os.path.dirname(__file__)


with open(f'{cwd}/utils/taggingEfficienciesDenseLookup.pkl', 'rb') as _file:
    taggingEffLookup = pickle.load(_file)

puLookup = util.load(f'{cwd}/ScaleFactors/puLookup.coffea')
puLookup_Down = util.load(f'{cwd}/ScaleFactors/puLookup_Down.coffea')
puLookup_Up = util.load(f'{cwd}/ScaleFactors/puLookup_Up.coffea')


muSFFileList = [{'id'   : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
                 'iso'   : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                 'trig'  : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                 'scale' : 19.656062760/35.882515396},
                {'id'     : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
                 'iso'   : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                 'trig'  : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                 'scale' : 16.226452636/35.882515396}]

Jetext = extractor()
Jetext.add_weight_sets([
        f"* * {cwd}/ScaleFactors/JEC/Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs.jec.txt",
        f"* * {cwd}/ScaleFactors/JEC/Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs.jec.txt",
        f"* * {cwd}/ScaleFactors/JEC/Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs.junc.txt",
        f"* * {cwd}/ScaleFactors/JEC/Summer16_25nsV1_MC_PtResolution_AK4PFchs.jr.txt",
        f"* * {cwd}/ScaleFactors/JEC/Summer16_25nsV1_MC_SF_AK4PFchs.jersf.txt",
        ])
Jetext.finalize()
Jetevaluator = Jetext.make_evaluator()

jec_names = ['Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs','Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs']
junc_names = ['Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs']

jer_names = ['Summer16_25nsV1_MC_PtResolution_AK4PFchs']
jersf_names = ['Summer16_25nsV1_MC_SF_AK4PFchs']


JECcorrector = FactorizedJetCorrector(**{name: Jetevaluator[name] for name in jec_names})

JECuncertainties = JetCorrectionUncertainty(**{name:Jetevaluator[name] for name in junc_names})

JER = JetResolution(**{name:Jetevaluator[name] for name in jer_names})
JERsf = JetResolutionScaleFactor(**{name:Jetevaluator[name] for name in jersf_names})

Jet_transformer = JetTransformer(jec=JECcorrector,junc=JECuncertainties, jer = JER, jersf = JERsf)


@numba.jit(nopython=True)
def maxHistoryPDGID(idxList_contents, idxList_starts, idxList_stops, pdgID_contents, pdgID_starts, pdgID_stops, motherIdx_contents, motherIdx_starts, motherIdx_stops):
    maxPDGID_array = np.ones(len(idxList_starts),np.int32)*-1
    for i in range(len(idxList_starts)):
        if idxList_starts[i]==idxList_stops[i]:
            continue
            
        idxList = idxList_contents[idxList_starts[i]:idxList_stops[i]]
        pdgID = pdgID_contents[pdgID_starts[i]:pdgID_stops[i]]
        motherIdx = motherIdx_contents[motherIdx_starts[i]:motherIdx_stops[i]]
    
        idx = idxList[0]
        maxPDGID = -1
        while idx>-1:
            pdg = pdgID[idx]
            maxPDGID = max(maxPDGID, abs(pdg))
            idx = motherIdx[idx]
        maxPDGID_array[i] = maxPDGID
    return maxPDGID_array



# Look at ProcessorABC to see the expected methods and what they are supposed to do
class TTGammaProcessor(processor.ProcessorABC):
#     def __init__(self, runNum = -1, eventNum = -1):
    def __init__(self, runNum = -1, eventNum = -1, mcEventYields = None):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################

        self.mcEventYields = mcEventYields

        dataset_axis = hist.Cat("dataset", "Dataset")
        lep_axis = hist.Cat("lepFlavor", "Lepton Flavor")

        systematic_axis = hist.Cat("systematic", "Systematic Uncertainty")

        # lep_axis = hist.Bin("lepFlavor", r"ElectronOrMuon", 2, -1, 1)
        # lep_axis.identifiers()[0].label = 'Electron'
        # lep_axis.identifiers()[1].label = 'Muon'

        m3_axis = hist.Bin("M3", r"$M_3$ [GeV]", 200, 0., 1000)
        mass_axis = hist.Bin("mass", r"$m_{\ell\gamma}$ [GeV]", 400, 0., 400)
        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 200, 0., 1000)
        eta_axis = hist.Bin("eta", r"$\eta_{\gamma}$", 300, -1.5, 1.5)
        chIso_axis = hist.Bin("chIso", r"Charged Hadron Isolation", np.arange(-0.1,20.001,.05))

        ## Define axis to keep track of photon category
        phoCategory_axis = hist.Bin("category", r"Photon Category", [1,2,3,4,5])
        phoCategory_axis.identifiers()[0].label = "Genuine Photon"    
        phoCategory_axis.identifiers()[1].label = "Misidentified Electron"    
        phoCategory_axis.identifiers()[2].label = "Hadronic Photon"    
        phoCategory_axis.identifiers()[3].label = "Hadronic Fake"    
        
        ###
        self._accumulator = processor.dict_accumulator({
            ##photon histograms
            'pho_pt': hist.Hist("Counts", dataset_axis, pt_axis),
            'photon_pt': hist.Hist("Counts", dataset_axis, pt_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_eta': hist.Hist("Counts", dataset_axis, eta_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_chIso': hist.Hist("Counts", dataset_axis, chIso_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_chIsoSideband': hist.Hist("Counts", dataset_axis, chIso_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_lepton_mass': hist.Hist("Counts", dataset_axis, mass_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_lepton_mass_3j0t': hist.Hist("Counts", dataset_axis, mass_axis, phoCategory_axis, lep_axis, systematic_axis),
            'M3'      : hist.Hist("Counts", dataset_axis, m3_axis, phoCategory_axis, lep_axis, systematic_axis),
            'M3Presel': hist.Hist("Counts", dataset_axis, m3_axis, lep_axis, systematic_axis),
            'EventCount':processor.value_accumulator(int)
        })

        self.eventNum = eventNum
        self.runNum = runNum

        ext = extractor()
        ext.add_weight_sets([f"btag2016 * {cwd}/ScaleFactors/Btag/DeepCSV_2016LegacySF_V1.btag.csv"])
        ext.finalize()
        self.evaluator = ext.make_evaluator()
        
        ele_id_file = uproot.open(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/ele2016/2016LegacyReReco_ElectronTight_Fall17V2.root')
        self.ele_id_sf = dense_lookup.dense_lookup(ele_id_file["EGamma_SF2D"].values, ele_id_file["EGamma_SF2D"].edges)
        self.ele_id_err = dense_lookup.dense_lookup(ele_id_file["EGamma_SF2D"].variances**0.5, ele_id_file["EGamma_SF2D"].edges)

        ele_reco_file = uproot.open(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/ele2016/egammaEffi.txt_EGM2D_runBCDEF_passingRECO.root')
        self.ele_reco_sf = dense_lookup.dense_lookup(ele_reco_file["EGamma_SF2D"].values, ele_reco_file["EGamma_SF2D"].edges)
        self.ele_reco_err = dense_lookup.dense_lookup(ele_reco_file["EGamma_SF2D"].variances**.5, ele_reco_file["EGamma_SF2D"].edges)

        
        
        
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

        self.mu_id_sf = dense_lookup.dense_lookup(mu_id_vals, mu_id_edges)
        self.mu_id_err = dense_lookup.dense_lookup(mu_id_err, mu_id_edges)
        self.mu_iso_sf = dense_lookup.dense_lookup(mu_iso_vals, mu_iso_edges)
        self.mu_iso_err = dense_lookup.dense_lookup(mu_iso_err, mu_iso_edges)
        self.mu_trig_sf = dense_lookup.dense_lookup(mu_trig_vals, mu_trig_edges)
        self.mu_trig_err = dense_lookup.dense_lookup(mu_trig_err, mu_trig_edges)
        

        
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        output['EventCount'] = len(events)

        datasetFull = events.metadata['dataset']
        dataset=datasetFull.replace('_2016','')
        isData = 'Data' in dataset
        
        rho = events.fixedGridRhoFastjetAll

###To-do list:
## Fix buggy overlap removal
## Jet transformer
## LHE and gen weights
      
        """
        #ARH: Fix jet variables, genJetIdx
     #       genJetIdx=df['Jet_genJetIdx'] if not isData else np.ones_like(df['Jet_jetId']),
     #       ptGenJet=np.zeros_like(df['Jet_pt']), #df['Jet_genJetIdx'] if not isData else np.ones_like(df['Jet_jetId']),
      
    #ARH    events["Jet","hadFlav"] = events.Jet.hadronFlavour if not isData else np.ones_like(events.Jet.jetId)
    #ARH    events["Jet","ptGenJet"]= np.zeros_like(events.Jet.pt)

        
        if not isData:
            #ARH: Not confident the changes below will work
            # fix what seems to be a bug, genJets get skimmed after the genJet matching:
            #   jets matched to a genJet with pt<10 still get assigned a value for Jet_genJetIdx, but that index is not present in the
            #   genJet list because it is cut.  In these cases, set idx to -1

            #ARH might have to use np.where or ak.where or something
#            events.Jet.genJetIdx[events.Jet.genJetIdx >= ak.num(events.GenJet)] = -1
#            events.Jet.ptGenJet[events.Jet.genJetIdx>-1] = events.Jet.matched_gen.pt
 
            events["Jets","rho"] = events.Jet.pt.ones_like()*rho

            Jet_transformer.transform(jets, forceStochastic=False)

        # *** ARH: tried to fix these variables; see if it worked
        chIso=df['Photon_pfRelIso03_chg']*df['Photon_pt']
        events["Photon","genFlav"]=events.Photon.genPartFlav if not isData else np.ones_like(events.Photon.electronVeto)
        events["Photon","genIdx"] =events.Photon.genPartIdx  if not isData else np.ones_like(events.Photon.electronVeto)
        # *** 
        
        if not isData:
            genmotherIdx = events.GenPart.genPartIdxMother
            genpdgid = events.GenPart.pdgId
            hasWeights=True

            try:
                generatorWeight = df['Generator_weight']
                generatorWeight.shape = (generatorWeight.size,1)

                LHEWeight_originalXWGTUP = df['LHEWeight_originalXWGTUP']
                LHEWeight_originalXWGTUP.shape = (LHEWeight_originalXWGTUP.size,1)

                nPSWeights = df['nPSWeight']
                PSWeights = df['PSWeight']
                PSWeights.shape = (nPSWeights.size,int(nPSWeights.mean()))
                if nPSWeights.mean()==1:
                    hasWeights=False
                
                nLHEScaleWeights = df['nLHEScaleWeight']
                LHEScaleWeights = df['LHEScaleWeight']
                LHEScaleWeights.shape = (nLHEScaleWeights.size,int(nLHEScaleWeights.mean()))
                
                nLHEPdfWeights = df['nLHEPdfWeight']
                LHEPdfWeights = df['LHEPdfWeight']
                LHEPdfWeights.shape = (nLHEPdfWeights.size,int(nLHEPdfWeights.mean()))


                #avoid errors from 0/0 division
                if (LHEPdfWeights[:,:1]==0).any():
                    LHEPdfWeights[:,0][LHEPdfWeights[:,0]==0] = 1.
                LHEPdfVariation = LHEPdfWeights / LHEPdfWeights[:,:1]

                if nLHEScaleWeights.mean()==9:
                    scaleWeightSelector=[0,1,3,5,7,8]
                elif nLHEScaleWeights.mean()==44:
                    scaleWeightSelector=[0,5,15,24,34,39]
                else:
                    scaleWeightSelector=[]

                LHEScaleVariation = LHEScaleWeights[:,scaleWeightSelector]

                if not (generatorWeight==LHEWeight_originalXWGTUP).all():
                    PSWeights = PSWeights * LHEWeight_originalXWGTUP / generatorWeight
            except:
                hasWeights=False
        """
        #ARH probably buggy, because phi might not be in range of [-pi, pi]
        def dR(one,two):
            pairs = ak.cartesian({"i0":one,"i1":two},nested=True)
            diffPhi = abs(pairs.i0.phi - pairs.i1.phi)
            diffEta = abs(pairs.i0.eta - pairs.i1.eta)
            return np.sqrt(diffPhi*diffPhi + diffEta*diffEta)



        #################
        # OVERLAP REMOVAL
        #################
        # Overlap removal between related samples
        # TTGamma and TTbar
        # WGamma and WJets
        # ZGamma and ZJets
        # We need to remove events from TTbar which are already counted in the phase space in which the TTGamma sample is produced
        # photon with pT> 10 GeV, eta<5, and at least dR>0.1 from other gen objects 
        doOverlapRemoval = False
        if 'TTbar' in dataset:
            doOverlapRemoval = True
            overlapPt = 10.
            overlapEta = 5.
            overlapDR = 0.1
        if re.search("^W[1234]jets$", dataset):
            doOverlapRemoval = True
            overlapPt = 10.
            overlapEta = 2.5
            overlapDR = 0.05
        if 'DYjetsM' in dataset:
            doOverlapRemoval = True
            overlapPt = 15.
            overlapEta = 2.6
            overlapDR = 0.05

            
        if doOverlapRemoval:
            genmotherIdx = events.GenPart.genPartIdxMother
            genpdgid = events.GenPart.pdgId

            overlapPhoSelect = ((events.GenPart.pt>=overlapPt) & 
                                (abs(events.GenPart.eta) < overlapEta) & 
                                (events.GenPart.pdgId==22) & 
                                (events.GenPart.status==1)
                               )
            #potential overlap photons are only those passing the kinematic cuts
            OverlapPhotons = events.GenPart[overlapPhoSelect] 

            #if the overlap photon is actually from a non prompt decay, it's not part of the phase space of the separate sample
           #ARH idx = OverlapPhotons.genPartIdxMother
            #ARH fix this: maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
            #                            genpdgid.content, genpdgid.starts, genpdgid.stops, 
            #                            genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)
           
            finalGen = events.GenPart[((events.GenPart.status==1)|(events.GenPart.status==71)) & 
                                      ~((abs(events.GenPart.pdgId)==12) | (abs(events.GenPart.pdgId)==14) | (abs(events.GenPart.pdgId)==16))]

#            genPairs = ak.cartesian({"pho":OverlapPhotons,"gen":finalGen},nested=True)
#ARH: need to remove cases where the cross product is the gen photon with itself            genPairs = genPairs[~(genPairs.pho == genPairs.gen)]
#            dRPairs  = genPairs.pho.delta_r(genPairs.gen)

            dRPairs = dR(OverlapPhotons,finalGen)
#            genPairs = OverlapPhotons['p4'].cross(finalGen['p4'],nested=True)
            ##remove the case where the cross produce is the gen photon with itself
#            genPairs = genPairs[~(genPairs.i0==genPairs.i1)]
#            #find closest gen particle to overlap photons
#            dRPairs = genPairs.i0.delta_r(genPairs.i1)


            #the event is overlapping with the separate sample if there is an overlap photon passing the dR cut and not coming from hadronic activity
            isOverlap = ak.any(( (ak.min(dRPairs,axis=-1)>overlapDR)  ), axis=-1)
            #ARH isOverlap = ak.any(( (ak.min(dRPairs,axis=-1)>overlapDR) & maxParent>37  ), axis=-1)
#((dRPairs.min()>overlapDR) & (maxParent<37)).any()
            passOverlapRemoval = ~isOverlap

        else:
            passOverlapRemoval = np.ones_like(len(events))==1
            
        
        ##################
        # OBJECT SELECTION
        ##################
         # PART 1A Uncomment to add in object selection
         
        # 1. ADD SELECTION

        #select tight muons
        # tight muons should have a pt of at least 30 GeV, |eta| < 2.4, pass the tight muon ID cut (tightID variable), and have a relative isolation of less than 0.15
        muonSelectTight = ((events.Muon.pt>30) & 
                           (abs(events.Muon.eta)<2.4) & 
                           (events.Muon.tightId) & 
                           (events.Muon.pfRelIso04_all < 0.15)
                          )

        #select loose muons        
        muonSelectLoose = ((events.Muon.pt>15) & 
                           (abs(events.Muon.eta)<2.4) & 
                           ((events.Muon.isPFcand) & (events.Muon.isTracker | events.Muon.isGlobal)) & 
                           (events.Muon.pfRelIso04_all < 0.25) &
                           np.invert(muonSelectTight)
                          )

        #define electron cuts
        eleEtaGap = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
        elePassDXY = ((abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dxy) < 0.05) |
                     (abs(events.Electron.eta) > 1.479)  & (abs(events.Electron.dxy) < 0.1)
                    )
        elePassDZ = ((abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dz) < 0.1) |
                     (abs(events.Electron.eta) > 1.479)  & (abs(events.Electron.dz) < 0.2)
                    )

        
        #select tight electrons
        # 1. ADD SELECTION
        #select tight electrons
        # tight electrons should have a pt of at least 35 GeV, |eta| < 2.1, pass the cut based electron id (cutBased variable in NanoAOD>=4), and pass the etaGap, D0, and DZ cuts defined above
        electronSelectTight = ((events.Electron.pt>35) & 
                               (abs(events.Electron.eta)<2.1) & 
                               eleEtaGap &      
                               (events.Electron.cutBased>=4) &
                               elePassDXY & 
                               elePassDZ
                              )

        #select loose electrons
        electronSelectLoose = ((events.Electron.pt>15) & 
                               (abs(events.Electron.eta)<2.4) & 
                               eleEtaGap &      
                               (events.Electron.cutBased>=1) &
                               elePassDXY & 
                               elePassDZ & 
                               np.invert(electronSelectTight)
                              )
        
        # 1. ADD SELECTION
        #  Object selection
        #select the subset of muons passing the muonSelectTight and muonSelectLoose cuts
        tightMuon = events.Muon[muonSelectTight]
        looseMuon = events.Muon[muonSelectLoose]
        
        # 1. ADD SELECTION
        #  Object selection
        #select the subset of electrons passing the electronSelectTight and electronSelectLoose cuts
        tightElectron = events.Electron[electronSelectTight]
        looseElectron = events.Electron[electronSelectLoose]

        #### Calculate deltaR between photon and nearest lepton 
        phoMu = dR(events.Photon, tightMuon)
        dRphomu = ak.all(phoMu>0.4, axis=-1) | (ak.num(tightMuon)==0)  
        
        phoEle = dR(events.Photon, tightElectron)
        dRphoele = ak.all(phoEle>0.4, axis=-1) | (ak.num(tightElectron)==0)

        #photon selection (no ID requirement used here)
        photonSelect = ((events.Photon.pt>20) & 
                        (abs(events.Photon.eta) < 1.4442) &
                        (events.Photon.isScEtaEE | events.Photon.isScEtaEB) &
                        (events.Photon.electronVeto) & 
                        np.invert(events.Photon.pixelSeed) & 
                        dRphomu & dRphoele
                       )
        

        #split out the ID requirement, enabling Iso and SIEIE to be inverted for control regions
        photonID = events.Photon.cutBased >= 2

        #parse VID cuts, define loose photons (not used yet)
        photon_MinPtCut = (events.Photon.vidNestedWPBitmap>>0 & 3)>=2 
        photon_PhoSCEtaMultiRangeCut = (events.Photon.vidNestedWPBitmap>>2 & 3)>=2 
        photon_PhoSingleTowerHadOverEmCut = (events.Photon.vidNestedWPBitmap>>4 & 3)>=2  
        photon_PhoFull5x5SigmaIEtaIEtaCut = (events.Photon.vidNestedWPBitmap>>6 & 3)>=2  
        photon_ChIsoCut = (events.Photon.vidNestedWPBitmap>>8 & 3)>=2  
        photon_NeuIsoCut = (events.Photon.vidNestedWPBitmap>>10 & 3)>=2  
        photon_PhoIsoCut = (events.Photon.vidNestedWPBitmap>>12 & 3)>=2  
        
        photonID_NoChIsoSIEIE = (photon_MinPtCut & 
                                 photon_PhoSCEtaMultiRangeCut & 
                                 photon_PhoSingleTowerHadOverEmCut & 
                                 photon_PhoFull5x5SigmaIEtaIEtaCut & 
                                 photon_NeuIsoCut & 
                                 photon_PhoIsoCut)

        # 1. ADD SELECTION
        #  Object selection
        #select tightPhotons, the subset of photons passing the photonSelect cut and the photonID cut        
        tightPhotons = events.Photon[photonSelect & photonID]
        #select loosePhotons, the subset of photons passing the photonSelect cut and all photonID cuts without the charged hadron isolation cut applied
        loosePhotons = events.Photon[photonSelect & photonID_NoChIsoSIEIE & photon_PhoFull5x5SigmaIEtaIEtaCut]

        #ARH: not sure if this is used
        loosePhotonsSideband = events.Photon[photonSelect & photonID_NoChIsoSIEIE & (events.Photon.sieie>0.012)]


        #ARH ADD JET TRANSFORMER HERE!

        ##check dR jet,lepton & jet,photon
        jetMu = dR(events.Jet,tightMuon)
        dRjetmu = ak.all(jetMu>0.4,axis=-1) | (ak.num(tightMuon)==0)

        jetEle = dR(events.Jet,tightElectron)
        dRjetele = ak.all(jetEle>0.4,axis=-1) | (ak.num(tightElectron)==0)

        jetPho = dR(events.Jet,tightPhotons)
        dRjetpho = ak.all(jetPho>0.4,axis=-1) | (ak.num(tightPhotons)==0)

        # 1. ADD SELECTION
        #select good jets
        # jetsshould have a pt of at least 30 GeV, |eta| < 2.4, pass the medium jet id (bit-wise selected from the jetID variable), and pass the delta R cuts defined above (dRjetmu, dRjetele, dRjetpho)
        ##medium jet ID cut                                                                                                                                            
        jetIDbit = 1

        jetSelectNoPt = ((abs(events.Jet.eta) < 2.4) &
                         ((events.Jet.jetId >> jetIDbit & 1)==1) &
                         dRjetmu & dRjetele & dRjetpho )
        
        jetSelect = jetSelectNoPt & (events.Jet.pt > 30)

        # 1. ADD SELECTION
        #select the subset of jets passing the jetSelect cuts
        tightJets = events.Jet[jetSelect]                                                                                                                             
    
        # 1. ADD SELECTION
        # select the subset of tightJets which pass the Deep CSV tagger
        bTagWP = 0.6321   #2016 DeepCSV working point
        btagged = events.Jet.btagDeepB>bTagWP  
        bTaggedJets= events.Jet[jetSelect & btagged]


        #####################
        # EVENT SELECTION
        #####################
        ### PART 1B: Uncomment to add event selection
        # 1. ADD SELECTION
        ## apply triggers
        # muon events should be triggered by either the HLT_IsoMu24 or HLT_IsoTkMu24 triggers
        # electron events should be triggered by HLT_Ele27_WPTight_Gsf trigger
        # HINT: trigger values can be accessed with the variable events.HLT.TRIGGERNAME, 
        # the bitwise or operator can be used to select multiple triggers events.HLT.TRIGGER1 | events.HLT.TRIGGER2
        muTrigger  = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
        eleTrigger = events.HLT.Ele27_WPTight_Gsf

        # 1. ADD SELECTION
        #  Event selection
        #oneMuon, should be true if there is exactly one tight muon in the event 
        # (hint, the ak.num() method returns the number of objects in each row of a jagged array)
        oneMuon = (ak.num(tightMuon) == 1)
        #muVeto, should be true if there are no tight muons in the event
        muVeto  = (ak.num(tightMuon) == 0)
        
        # 1. ADD SELECTION
        #  Event selection
 
        #oneEle should be true if there is exactly one tight electron in the event
        oneEle  = (ak.num(tightElectron) == 1)
    
        #eleVeto should be true if there are no tight electrons in the event
        eleVeto = (ak.num(tightElectron) == 0)

        # 1. ADD SELECTION
        #  Event selection
        #looseMuonVeto and looseElectronVeto should be true if there are 0 loose muons or electrons in the event
        looseMuonVeto = (ak.num(looseMuon) == 0)
        looseElectronVeto = (ak.num(looseElectron) == 0)

        # 1. ADD SELECTION
        # muon selection, requires events to pass:   muon trigger
        #                                            overlap removal
        #                                            have exactly one muon
        #                                            have no electrons
        #                                            have no loose muons
        #                                            have no loose electrons
        #ARH uncomment passOverlapRemoval
        muon_eventSelection = (muTrigger & #passOverlapRemoval & 
                               oneMuon & eleVeto & 
                               looseMuonVeto & looseElectronVeto) 

        # electron selection, requires events to pass:   electron trigger
        #                                                overlap removal
        #                                                have exactly one electron
        #                                                have no muons
        #                                                have no loose muons
        #                                                have no loose electrons
        #ARH uncomment passOverlapRemoval 
        electron_eventSelection = (eleTrigger & #passOverlapRemoval &
                                   oneEle & muVeto & 
                                   looseMuonVeto & looseElectronVeto)  

        # 1. ADD SELECTION
        #add selection 'eleSel', for events passing the electron event selection, and muSel for those passing the muon event selection
        #  ex: selection.add('testSelection', array_of_booleans)
    
        #create a selection object
        selection = processor.PackedSelection()

        selection.add('eleSel',ak.to_numpy(electron_eventSelection))
        selection.add('muSel',ak.to_numpy(muon_eventSelection))

        #add two jet selection criteria
        #   First, 'jetSel' which selects events with at least 4 tightJets and at least one bTaggedJets
        nJets = 4
        selection.add('jetSel',     ak.to_numpy( (ak.num(tightJets) >= nJets) & (ak.num(bTaggedJets) >= 1) ))
        #   Second, 'jetSel_3j0t' which selects events with at least 3 tightJets and exactly zero bTaggedJets
        selection.add('jetSel_3j0t', ak.to_numpy((ak.num(tightJets) >= 3)     & (ak.num(bTaggedJets) == 0) ))

        # add selection for events with exactly 0 tight photons
        selection.add('zeroPho', ak.to_numpy(ak.num(tightPhotons) == 0))

        # add selection for events with exactly 1 tight photon
        selection.add('onePho',  ak.to_numpy(ak.num(tightPhotons) == 1))

        # add selection for events with exactly 1 loose photon
        selection.add('loosePho',ak.to_numpy(ak.num(loosePhotons) == 1))

        #ARH not sure we use this anymore
        #add selection for events with exactly 1 loose photon from the sideband
        selection.add('loosePhoSideband', ak.to_numpy(ak.num(loosePhotonsSideband) == 1))

        ##################
        # EVENT VARIABLES
        ##################

        # PART 2A: Uncomment to begin implementing event variables
        # 2. DEFINE VARIABLES
        ## Define M3, mass of 3-jet pair with highest pT
        # find all possible combinations of 3 tight jets in the events 
        #hint: using the ak.combinations(array,n) method chooses n unique items from array. Use the "fields" option to define keys you can use to access the items
        triJet=ak.combinations(tightJets,3,fields=["first","second","third"])
        triJetPt = (triJet.first + triJet.second + triJet.third).pt
        triJetMass = (triJet.first + triJet.second + triJet.third).mass
        M3 = triJetMass[ak.argmax(triJetPt,axis=-1)]

        leadingMuon = tightMuon[::1]
        leadingElectron = tightElectron[::1]

        leadingPhoton = tightPhotons[:,:1]
        leadingPhotonLoose = loosePhotons[:,:1]
        leadingPhotonSideband = loosePhotonsSideband[:,:1]

        # 2. DEFINE VARIABLES
        # define egammaMass, mass of combinations of tightElectron and leadingPhoton (hint: using the .cross() method)
        egammaPairs = ak.cartesian({"i0":tightElectron,"i1":leadingPhoton},nested=True)
        #ARH can't figure this one out yet
        """
        egammaMass  = ?
        # define egammaMass, mass of combinations of tightElectron and leadingPhoton (hint: using the .cross() method)
        mugammaPairs = ?
        mugammaMass = ?
        egamma = leadingElectron['p4'].cross(leadingPhoton['p4'])
        mugamma = leadingMuon['p4'].cross(leadingPhoton['p4'])
        egammaMass = (egamma.i0 + egamma.i1).mass
        mugammaMass = (mugamma.i0 + mugamma.i1).mass
        """

        ###################
        # PHOTON CATEGORIES
        ###################

        #ARH: this is how far I got pre-Christmas
        # Define photon category for each event
        """
        if not isData:
            #### Photon categories, using genIdx branch
            # reco photons really generated as electrons
            idx = leadingPhoton.genIdx

            matchedPho = (genpdgid[idx]==22).any()
            isMisIDele = (abs(genpdgid[idx])==11).any()
            
            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            hadronicParent = maxParent>25

            isGenPho = matchedPho & ~hadronicParent
            isHadPho = matchedPho & hadronicParent
            isHadFake = ~(isMisIDele | isGenPho | isHadPho) & (ak.num(leadingPhoton)==1)
            
            #define integer definition for the photon category axis
            phoCategory = 1*isGenPho + 2*isMisIDele + 3*isHadPho + 4*isHadFake
            

            isMisIDeleLoose = (leadingPhotonLoose.genFlav==13).any()
            matchedPhoLoose = (leadingPhotonLoose.genFlav==1).any()

            # look through parentage to find if any hadrons in genPhoton parent history
            idx = leadingPhotonLoose.genIdx

            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            hadronicParent = maxParent>25

            isGenPhoLoose = matchedPhoLoose & ~hadronicParent
            isHadPhoLoose = matchedPhoLoose & hadronicParent
            isHadFakeLoose = ~(isMisIDeleLoose | isGenPhoLoose | isHadPhoLoose) & (ak.num(leadingPhotonLoose)==1)        

            #define integer definition for the photon category axis
            phoCategoryLoose = 1*isGenPhoLoose + 2*isMisIDeleLoose + 3*isHadPhoLoose + 4*isHadFakeLoose

            
            isMisIDeleSideband = (leadingPhotonSideband.genFlav==13).any()
            matchedPhoSideband = (leadingPhotonSideband.genFlav==1).any()

            # look through parentage to find if any hadrons in genPhoton parent history
            idx = leadingPhotonSideband.genIdx

            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            hadronicParent = maxParent>25

            isGenPhoSideband = matchedPhoSideband & ~hadronicParent
            isHadPhoSideband = matchedPhoSideband & hadronicParent
            isHadFakeSideband = ~(isMisIDeleSideband | isGenPhoSideband | isHadPhoSideband) & (ak.num(leadingPhotonSideband)==1)        

            #define integer definition for the photon category axis
            phoCategorySideband = 1*isGenPhoSideband + 2*isMisIDeleSideband + 3*isHadPhoSideband + 4*isHadFakeSideband            
        else:
            phoCategory = np.ones(df.size)
            phoCategoryLoose = np.ones(df.size)
            phoCategorySideband = np.ones(df.size)
        
        if not isData:
            selection.add('jetSel_JERUp', (ak.num(tightJets_JERUp) >= nJets) & ((tightJets_JERUp.btagDeepB>bTagWP).sum() >= 1) )
            selection.add('jetSel_JERUp_3j0t', (ak.num(tightJets_JERUp) >= 3) & ((tightJets_JERUp.btagDeepB>bTagWP).sum() == 0) )
            
            selection.add('jetSel_JERDown', (ak.num(tightJets_JERDown) >= nJets) & ((tightJets_JERDown.btagDeepB>bTagWP).sum() >= 1) )
            selection.add('jetSel_JERDown_3j0t', (ak.num(tightJets_JERDown) >= 3) & ((tightJets_JERDown.btagDeepB>bTagWP).sum() == 0) )
            
            selection.add('jetSel_JESUp', (ak.num(tightJets_JESUp) >= nJets) & ((tightJets_JESUp.btagDeepB>bTagWP).sum() >= 1) )
            selection.add('jetSel_JESUp_3j0t', (ak.num(tightJets_JESUp) >= 3) & ((tightJets_JESUp.btagDeepB>bTagWP).sum() == 0) )
            
            selection.add('jetSel_JESDown', (ak.num(tightJets_JESDown) >= nJets) & ((tightJets_JESDown.btagDeepB>bTagWP).sum() >= 1) )
            selection.add('jetSel_JESDown_3j0t', (ak.num(tightJets_JESDown) >= 3) & ((tightJets_JESDown.btagDeepB>bTagWP).sum() == 0) )

        """
        ################
        # EVENT WEIGHTS
        ################

        #create a processor Weights object, with the same length as the number of events in the chunk
        weights = processor.Weights(len(events))
  
        """
#        evtWeight = np.ones_like(df['event'],dtype=np.float64)        
        if not 'Data' in dataset:
            lumiWeight = np.ones(df.size)
            nMCevents = self.mcEventYields[datasetFull]
            xsec = crossSections[dataset]
            lumiWeight *= xsec * lumis[year] / nMCevents

#            evtWeight *= xsec * lumis[year] / nMCevents

            weights.add('lumiWeight',lumiWeight)

            nPUTrue = df['Pileup_nTrueInt']
            puWeight = puLookup(datasetFull, nPUTrue)
            puWeight_Up = puLookup_Up(datasetFull, nPUTrue)
            puWeight_Down = puLookup_Down(datasetFull, nPUTrue)
            
            weights.add('puWeight',weight=puWeight, weightUp=puWeight_Up, weightDown=puWeight_Down)

            #btag key name
            #name / working Point / type / systematic / jetType
            #  ... / 0-loose 1-medium 2-tight / comb,mujets,iterativefit / central,up,down / 0-b 1-c 2-udcsg 

            bJetSF_b = self.evaluator['btag%iDeepCSV_1_comb_central_0'%year](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btagDeepB)
            bJetSF_c = self.evaluator['btag%iDeepCSV_1_comb_central_1'%year](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btagDeepB)
            bJetSF_udcsg = self.evaluator['btag%iDeepCSV_1_incl_central_2'%year](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btagDeepB)

            bJetSF_b_up = self.evaluator['btag%iDeepCSV_1_comb_up_0'%year](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btagDeepB)
            bJetSF_c_up = self.evaluator['btag%iDeepCSV_1_comb_up_1'%year](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btagDeepB)
            bJetSF_udcsg_up = self.evaluator['btag%iDeepCSV_1_incl_up_2'%year](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btagDeepB)

            bJetSF_b_down = self.evaluator['btag%iDeepCSV_1_comb_down_0'%year](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btagDeepB)
            bJetSF_c_down = self.evaluator['btag%iDeepCSV_1_comb_down_1'%year](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btagDeepB)
            bJetSF_udcsg_down = self.evaluator['btag%iDeepCSV_1_incl_down_2'%year](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btagDeepB)

            bJetSF = JaggedArray(content = np.ones_like(tightJets.pt.content,dtype=np.float64), starts = tightJets.starts, stops = tightJets.stops)
            bJetSF.content[(tightJets.hadFlav==5).content] = bJetSF_b.content
            bJetSF.content[(tightJets.hadFlav==4).content] = bJetSF_c.content
            bJetSF.content[(tightJets.hadFlav==0).content] = bJetSF_udcsg.content

            bJetSF_heavy_up = JaggedArray(content = np.ones_like(tightJets.pt.content,dtype=np.float64), starts = tightJets.starts, stops = tightJets.stops)
            bJetSF_heavy_up.content[(tightJets.hadFlav==5).content] = bJetSF_b_up.content
            bJetSF_heavy_up.content[(tightJets.hadFlav==4).content] = bJetSF_c_up.content
            bJetSF_heavy_up.content[(tightJets.hadFlav==0).content] = bJetSF_udcsg.content

            bJetSF_heavy_down = JaggedArray(content = np.ones_like(tightJets.pt.content,dtype=np.float64), starts = tightJets.starts, stops = tightJets.stops)
            bJetSF_heavy_down.content[(tightJets.hadFlav==5).content] = bJetSF_b_down.content
            bJetSF_heavy_down.content[(tightJets.hadFlav==4).content] = bJetSF_c_down.content
            bJetSF_heavy_down.content[(tightJets.hadFlav==0).content] = bJetSF_udcsg.content

            bJetSF_light_up = JaggedArray(content = np.ones_like(tightJets.pt.content,dtype=np.float64), starts = tightJets.starts, stops = tightJets.stops)
            bJetSF_light_up.content[(tightJets.hadFlav==5).content] = bJetSF_b.content
            bJetSF_light_up.content[(tightJets.hadFlav==4).content] = bJetSF_c.content
            bJetSF_light_up.content[(tightJets.hadFlav==0).content] = bJetSF_udcsg_up.content

            bJetSF_light_down = JaggedArray(content = np.ones_like(tightJets.pt.content,dtype=np.float64), starts = tightJets.starts, stops = tightJets.stops)
            bJetSF_light_down.content[(tightJets.hadFlav==5).content] = bJetSF_b.content
            bJetSF_light_down.content[(tightJets.hadFlav==4).content] = bJetSF_c.content
            bJetSF_light_down.content[(tightJets.hadFlav==0).content] = bJetSF_udcsg_down.content

            ## mc efficiency lookup, data efficiency is eff* scale factor
            btagEfficiencies = taggingEffLookup(datasetFull,tightJets.hadFlav,tightJets.pt,tightJets.eta)
            btagEfficienciesData = btagEfficiencies*bJetSF

            btagEfficienciesData_b_up   = btagEfficiencies*bJetSF_heavy_up
            btagEfficienciesData_b_down = btagEfficiencies*bJetSF_heavy_down
            btagEfficienciesData_l_up   = btagEfficiencies*bJetSF_light_up
            btagEfficienciesData_l_down = btagEfficiencies*bJetSF_light_down

            ##probability is the product of all efficiencies of tagged jets, times product of 1-eff for all untagged jets
            ## https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
            pMC          = ak.prod(btagEfficiencies[btagged], axis=-1)           * ak.prod((1.-btagEfficiencies[np.invert(btagged)]), axis=-1) 
            pData        = ak.prod(btagEfficienciesData[btagged], axis=-1)       * ak.prod((1.-btagEfficienciesData[np.invert(btagged)]),axis=-1)
            pData_b_up   = ak.prod(btagEfficienciesData_b_up[btagged], axis=-1)  * ak.prod((1.-btagEfficienciesData_b_up[np.invert(btagged)]),axis=-1)
            pData_b_down = ak.prod(btagEfficienciesData_b_down[btagged],axis=-1) * ak.prod((1.-btagEfficienciesData_b_down[np.invert(btagged)]),axis=-1)
            pData_l_up   = ak.prod(btagEfficienciesData_l_up[btagged],axis=-1)   * ak.prod((1.-btagEfficienciesData_l_up[np.invert(btagged)]),axis=-1)
            pData_l_down = ak.prod(btagEfficienciesData_l_down[btagged],axis=-1) * ak.prod((1.-btagEfficienciesData_l_down[np.invert(btagged)]),axis=-1)

            pMC[pMC==0]=1. #avoid 0/0 error
            btagWeight = pData/pMC

            pData[pData==0] = 1. #avoid divide by 0 error
            btagWeight_b_up = pData_b_up/pData
            btagWeight_b_down = pData_b_down/pData
            btagWeight_l_up = pData_l_up/pData
            btagWeight_l_down = pData_l_down/pData

#            evtWeight *= btagWeight

            weights.add('btagWeight',btagWeight)

            weights.add('btagWeight_heavy',weight=np.ones_like(btagWeight), weightUp=btagWeight_b_up, weightDown=btagWeight_b_down)
            weights.add('btagWeight_light',weight=np.ones_like(btagWeight), weightUp=btagWeight_l_up, weightDown=btagWeight_l_down)

            eleID = self.ele_id_sf(tightElectron.eta, tightElectron.pt)
            eleIDerr = self.ele_id_err(tightElectron.eta, tightElectron.pt)
            eleRECO = self.ele_reco_sf(tightElectron.eta, tightElectron.pt)
            eleRECOerr = self.ele_reco_err(tightElectron.eta, tightElectron.pt)
            
            eleSF = ak.prod((eleID*eleRECO), axis=-1)
            eleSF_up   = ak.prod(((eleID + eleIDerr) * (eleRECO + eleRECOerr)), axis=-1)
            eleSF_down = ak.prod(((eleID - eleIDerr) * (eleRECO - eleRECOerr)), axis=-1)

            weights.add('eleEffWeight',weight=eleSF,weightUp=eleSF_up,weightDown=eleSF_down)

#            evtWeight *= eleSF

            muID = self.mu_id_sf(tightMuon.eta, tightMuon.pt)
            muIDerr = self.mu_id_err(tightMuon.eta, tightMuon.pt)
            muIso = self.mu_iso_sf(tightMuon.eta, tightMuon.pt)
            muIsoerr = self.mu_iso_err(tightMuon.eta, tightMuon.pt)
            muTrig = self.mu_iso_sf(abs(tightMuon.eta), tightMuon.pt)
            muTrigerr = self.mu_iso_err(abs(tightMuon.eta), tightMuon.pt)
            
            muSF = ak.prod((muID*muIso*muTrig), axis=-1)
            muSF_up   = ak.prod(((muID + muIDerr) * (muIso + muIsoerr) * (muTrig + muTrigerr)), axis=-1)
            muSF_down = ak.prod(((muID - muIDerr) * (muIso - muIsoerr) * (muTrig - muTrigerr)), axis=-1)

            weights.add('muEffWeight',weight=muSF,weightUp=muSF_up, weightDown=muSF_down)

            if hasWeights:
                weights.add('ISR',weight=np.ones(df.size), weightUp=PSWeights[:,2], weightDown=PSWeights[:,0])
                
                weights.add('FSR',weight=np.ones(df.size), weightUp=PSWeights[:,3], weightDown=PSWeights[:,1])

                weights.add('PDF', weight=np.ones(df.size), weightUp=LHEPdfVariation.max(axis=1), weightDown=LHEPdfVariation.min(axis=1))

                weights.add('Q2Scale', weight=np.ones(df.size), weightUp=LHEScaleVariation.max(axis=1), weightDown=LHEScaleVariation.min(axis=1))
            else:
                weights.add('ISR',    weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))
                weights.add('FSR',    weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))
                weights.add('PDF',    weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))
                weights.add('Q2Scale',weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))

#            evtWeight *= muSF
        """

        ###################
        # FILL HISTOGRAMS
        ###################
        # PART 3: Uncomment to add histograms
        systList = ['noweight','nominal']

        # PART 4: SYSTEMATICS
        # uncomment the full list after systematics have been implemented        
#        systList = ['noweight','nominal','puWeightUp','puWeightDown','muEffWeightUp','muEffWeightDown','eleEffWeightUp','eleEffWeightDown','btagWeight_lightUp','btagWeight_lightDown','btagWeight_heavyUp','btagWeight_heavyDown', 'ISRUp', 'ISRDown', 'FSRUp', 'FSRDown', 'PDFUp', 'PDFDown', 'Q2ScaleUp', 'Q2ScaleDown', 'JERUp', 'JERDown', 'JESUp', 'JESDown']




        if isData:
            systList = ['noweight']

        output['pho_pt'].fill(dataset=dataset,
                                 pt=ak.flatten(tightPhotons.pt[:,:1]))

        """
        for syst in systList:

            #find the event weight to be used when filling the histograms    
            weightSyst = syst

            #in the case of 'nominal', or the jet energy systematics, no weight systematic variation is used (weightSyst=None)
            if syst in ['nominal','JERUp','JERDown','JESUp','JESDown']:
                weightSyst=None
                
            if syst=='noweight':
                evtWeight = np.ones(len(events))
            else:
                # call weights.weight() with the name of the systematic to be varied
                evtWeight = weights.weight(weightSyst)

            jetSelType = 'jetSel'
            M3var = M3
            if syst in ['JERUp','JERDown','JESUp','JESDown']:
                jetSelType = f'jetSel_{syst}'
                M3var = eval(f'M3_{syst}')

            #loop over both electron and muon selections
            for lepton in ['electron','muon']:
                if lepton=='electron':
                    lepSel='eleSel'
                if lepton=='muon':
                    lepSel='muSel'

                phosel = selection.all(*(lepSel, jetSelType, 'onePho'))
                phoselLoose = selection.all(*(lepSel, jetSelType, 'loosePho') )
                phoselSideband = selection.all(*(lepSel, jetSelType, 'loosePhoSideband') )
                zeroPho = selection.all(*(lepSel, jetSelType, 'zeroPho') )

                output['photon_pt'].fill(dataset=dataset,
                                         pt=ak.flatten(tightPhotons.p4.pt[:,:1][phosel]),
                                         category=ak.flatten(phoCategory[phosel]),
                                         lepFlavor=lepton,
                                         systematic=syst,
                                         weight=evtWeight[phosel])
#                                         weight=evtWeight[phosel].flatten())
    
                output['photon_eta'].fill(dataset=dataset,
                                          eta=ak.flatten(tightPhotons.eta[:,:1][phosel]),
                                          category=ak.flatten(phoCategory[phosel]),
                                          lepFlavor=lepton,
                                          systematic=syst,
                                          weight=evtWeight[phosel])
 #                                         weight=evtWeight[phosel].flatten())
                
                output['photon_chIsoSideband'].fill(dataset=dataset,
                                                    chIso=ak.flatten(loosePhotonsSideband.chIso[:,:1][phoselSideband]),
                                                    category=ak.flatten(phoCategorySideband[phoselSideband]),
                                                    lepFlavor=lepton,
                                                    systematic=syst,
                                                    weight=evtWeight[phoselSideband])
#                                                    weight=evtWeight[phoselSideband].flatten())
                
                output['photon_chIso'].fill(dataset=dataset,
                                            chIso=ak.flatten(loosePhotons.chIso[:,:1][phoselLoose]),
                                            category=ak.flatten(phoCategoryLoose[phoselLoose]),
                                            lepFlavor=lepton,
                                            systematic=syst,
                                            weight=evtWeight[phoselLoose])
#                                            weight=evtWeight[phoselLoose].flatten())
                
                
                output['M3'].fill(dataset=dataset,
                                  M3=ak.flatten(M3var[phosel]),
                                  category=ak.flatten(phoCategoryLoose[phosel]),
                                  lepFlavor=lepton,
                                  systematic=syst,
                                  weight=evtWeight[phosel])
#                                  weight=evtWeight[phosel].flatten())
                
                output['M3Presel'].fill(dataset=dataset,
                                        M3=ak.flatten(M3var[zeroPho]),
                                        lepFlavor=lepton,
                                        systematic=syst,
                                        weight=evtWeight[zeroPho])
#                                        weight=evtWeight[zeroPho].flatten())                            
    
            
            phosel_e = selection.all(*('eleSel', jetSelType, 'onePho') )
            phosel_mu = selection.all(*('muSel', jetSelType, 'onePho') )

            phosel_3j0t_e = selection.all(*('eleSel', f'{jetSelType}_3j0t', 'onePho') )
            phosel_3j0t_mu = selection.all(*('muSel', f'{jetSelType}_3j0t', 'onePho') )

            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=ak.flatten(egammaMass[phosel_e]),
                                              category=ak.flatten(phoCategory[phosel_e]),
                                              lepFlavor='electron',
                                              systematic=syst,
                                              weight=evtWeight[phosel_e])
#                                              weight=evtWeight[phosel_e].flatten())
            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=ak.flatten(mugammaMass[phosel_mu]),
                                              category=ak.flatten(phoCategory[phosel_mu]),
                                              lepFlavor='muon',
                                              systematic=syst,
                                              weight=evtWeight[phosel_mu])
#                                              weight=evtWeight[phosel_mu].flatten())
    
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=ak.flatten(egammaMass[phosel_3j0t_e]),
                                                   category=ak.flatten(phoCategory[phosel_3j0t_e]),
                                                   lepFlavor='electron',
                                                   systematic=syst,
                                                   weight=evtWeight[phosel_3j0t_e])
#                                                   weight=evtWeight[phosel_3j0t_e].flatten())
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=ak.flatten(mugammaMass[phosel_3j0t_mu]),
                                                   category=ak.flatten(phoCategory[phosel_3j0t_mu]),
                                                   lepFlavor='muon',
                                                   systematic=syst,
                                                   weight=evtWeight[phosel_3j0t_mu])
#                                                   weight=evtWeight[phosel_3j0t_mu].flatten())
            


        """
        return output

    def postprocess(self, accumulator):
        return accumulator



