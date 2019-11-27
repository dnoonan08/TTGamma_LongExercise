import time

from coffea import hist, util
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from functools import partial
import uproot

from awkward import JaggedArray
import numpy as np
import pickle
import sys
from coffea.lookup_tools import extractor, dense_lookup
import numba
import re

from .utils.plotting import plotWithRatio
from .utils.crossSections import *
from .utils.efficiencies import getMuSF, getEleSF


import os.path
cwd = os.path.dirname(__file__)


with open(f'{cwd}/utils/taggingEfficienciesDenseLookup.pkl', 'rb') as _file:
    taggingEffLookup = pickle.load(_file)


muSFFileList = [{'id'   : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
                 'iso'   : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunBCDEF_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                 'trig'  : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunBtoF.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                 'scale' : 19.656062760/35.882515396},
                {'id'     : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ID.root", "NUM_TightID_DEN_genTracks_eta_pt"),
                 'iso'   : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_legacy_rereco_rootfiles_RunGH_SF_ISO.root", "NUM_TightRelIso_DEN_TightIDandIPCut_eta_pt"),
                 'trig'  : (f"{cwd}/ScaleFactors/MuEGammaScaleFactors/mu2016/EfficienciesStudies_2016_trigger_EfficienciesAndSF_RunGtoH.root", "IsoMu24_OR_IsoTkMu24_PtEtaBins/abseta_pt_ratio"),
                 'scale' : 16.226452636/35.882515396}]



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

    def process(self, df):
        output = self.accumulator.identity()

        datasetFull = df['dataset']
        dataset=datasetFull.replace('_2016','')

        isData = 'Data' in dataset
        
        year=2016
        yearStr="2016"
        muTrigger = df['HLT_IsoMu24'] | df['HLT_IsoTkMu24']
        eleTrigger = df['HLT_Ele27_WPTight_Gsf']
        photonBitMapName = 'Photon_cutBased'

        weights = processor.Weights(len(df['event']))
  
        #### These are already applied in the skim
#         filters = (df['Flag_goodVertices'] &
#                    df['Flag_globalSuperTightHalo2016Filter'] &
#                    df['Flag_HBHENoiseFilter'] &
#                    df['Flag_HBHENoiseIsoFilter'] &
#                    df['Flag_EcalDeadCellTriggerPrimitiveFilter'] &
#                    df['Flag_BadPFMuonFilter'] 
#                   )
#         if year > 2016:
#             filters = (filters & 
#                        df['Flag_ecalBadCalibFilterV2']
#                       )
        
        
        
        muons = JaggedCandidateArray.candidatesfromcounts(
            df['nMuon'],
            pt=df['Muon_pt'],
            eta=df['Muon_eta'],
            phi=df['Muon_phi'],
            mass=df['Muon_mass'],
            charge=df['Muon_charge'],
            relIso=df['Muon_pfRelIso04_all'],
            tightId=df['Muon_tightId'],
            isPFcand=df['Muon_isPFcand'],
            isTracker=df['Muon_isTracker'],
            isGlobal=df['Muon_isGlobal'],           
        )
        
        electrons = JaggedCandidateArray.candidatesfromcounts(
            df['nElectron'],
            pt=df['Electron_pt'],
            eta=df['Electron_eta'],
            phi=df['Electron_phi'],
            mass=df['Electron_mass'],
            charge=df['Electron_charge'],
            cutBased=df['Electron_cutBased'],
            d0=df['Electron_dxy'],
            dz=df['Electron_dz'],
        )

        jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'],
            eta=df['Jet_eta'],
            phi=df['Jet_phi'],
            mass=df['Jet_mass'],
            jetId=df['Jet_jetId'],
            btag=df['Jet_btagDeepB'],
            hadFlav=df['Jet_hadronFlavour'] if not isData else np.ones_like(df['Jet_jetId']),
            genIdx=df['Jet_genJetIdx'] if not isData else np.ones_like(df['Jet_jetId']),
        )

        photons = JaggedCandidateArray.candidatesfromcounts(
            df['nPhoton'],
            pt=df['Photon_pt'],
            eta=df['Photon_eta'],
            phi=df['Photon_phi'],
            mass=np.zeros_like(df['Photon_pt']),
            isEE=df['Photon_isScEtaEE'],
            isEB=df['Photon_isScEtaEB'],
            photonId=df[photonBitMapName],
            passEleVeto=df['Photon_electronVeto'],
            pixelSeed=df['Photon_pixelSeed'],
            sieie=df['Photon_sieie'],
            chIso=df['Photon_pfRelIso03_chg']*df['Photon_pt'],
            vidCuts=df['Photon_vidNestedWPBitmap'],
            genFlav=df['Photon_genPartFlav'] if not isData else np.ones_like(df['Photon_electronVeto']),
            genIdx=df['Photon_genPartIdx'] if not isData else np.ones_like(df['Photon_electronVeto']),
        )
        if not isData:
            genPart = JaggedCandidateArray.candidatesfromcounts(
                df['nGenPart'],
                pt=df['GenPart_pt'],
                eta=df['GenPart_eta'],
                phi=df['GenPart_phi'],
                mass=df['GenPart_mass'],
                pdgid=df['GenPart_pdgId'],
                motherIdx=df['GenPart_genPartIdxMother'],
                status=df['GenPart_status'],
                statusFlags=df['GenPart_statusFlags'],
            )

            genmotherIdx = genPart.motherIdx
            genpdgid = genPart.pdgid

        ## TTbar vs TTGamma Overlap Removal (work in progress, still buggy)
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
            overlapPhoSelect = ((genPart.pt>=overlapPt) & 
                                (abs(genPart.eta) < overlapEta) & 
                                (genPart.pdgid==22) & 
                                (genPart.status==1)
                               )
            
            OverlapPhotons = genPart[overlapPhoSelect] 

            idx = OverlapPhotons.motherIdx
            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)
            
            isNonPrompt = (maxParent>37).any()

            finalGen = genPart[((genPart.status==1)|(genPart.status==71)) & ~((abs(genPart.pdgid)==12) | (abs(genPart.pdgid)==14) | (abs(genPart.pdgid)==16))]

            genPairs = OverlapPhotons['p4'].cross(finalGen['p4'],nested=True)
            ##remove the case where the cross produce is the gen photon with itself
            genPairs = genPairs[~(genPairs.i0==genPairs.i1)]

            dRPairs = genPairs.i0.delta_r(genPairs.i1)
            
            isOverlap = ((dRPairs.min()>overlapDR) & (maxParent<37)).any()
            passOverlapRemoval = ~isOverlap
        else:
            passOverlapRemoval = np.ones_like(df['event'])==1
            


        
        muonSelectTight = ((muons.pt>30) & 
                           (abs(muons.eta)<2.4) & 
                           (muons.tightId) & 
                           (muons.relIso < 0.15)
                          )
        
        muonSelectLoose = ((muons.pt>15) & 
                           (abs(muons.eta)<2.4) & 
                           ((muons.isPFcand) & (muons.isTracker | muons.isGlobal)) & 
                           (muons.relIso < 0.25) &
                           np.invert(muonSelectTight)
                          )

        eleEtaGap = (abs(electrons.eta) < 1.4442) | (abs(electrons.eta) > 1.566)
        elePassD0 = ((abs(electrons.eta) < 1.479) & (abs(electrons.d0) < 0.05) |
                     (abs(electrons.eta) > 1.479)  & (abs(electrons.d0) < 0.1)
                    )
        elePassDZ = ((abs(electrons.eta) < 1.479) & (abs(electrons.dz) < 0.1) |
                     (abs(electrons.eta) > 1.479)  & (abs(electrons.dz) < 0.2)
                    )

        
        
        electronSelectTight = ((electrons.pt>35) & 
                               (abs(electrons.eta)<2.1) & 
                               eleEtaGap &      
                               (electrons.cutBased>=4) &
                               elePassD0 & 
                               elePassDZ
                              )

        electronSelectLoose = ((electrons.pt>15) & 
                               (abs(electrons.eta)<2.4) & 
                               eleEtaGap &      
                               (electrons.cutBased>=1) &
                               elePassD0 & 
                               elePassDZ & 
                               np.invert(electronSelectTight)
                              )
        
        tightMuon = muons[muonSelectTight]
        looseMuon = muons[muonSelectLoose]
        
        tightElectron = electrons[electronSelectTight]
        looseElectron = electrons[electronSelectLoose]


        
        oneMuon = (tightMuon.counts == 1)
        muVeto = (tightMuon.counts == 0)
        oneEle = (tightElectron.counts == 1)
        eleVeto = (tightElectron.counts == 0)
        looseMuonSel = (looseMuon.counts == 0)
        looseElectronSel = (looseElectron.counts == 0)

        
        #### Calculate deltaR between photon and nearest muon
        ####### make combination pairs
        phoMu = photons['p4'].cross(tightMuon['p4'],nested=True)
        
        ####### check delta R of each combination, if min is >0.1 it is okay, or if there are no tight muons it passes
        dRphomu = (phoMu.i0.delta_r(phoMu.i1)>0.4).all() | (tightMuon.counts==0)
        phoEle = photons['p4'].cross(tightElectron['p4'],nested=True)
        dRphoele = ((phoEle.i0.delta_r(phoEle.i1)).min()>0.4) | (tightElectron.counts==0)
        
        #photon selection (no ID requirement used here)
        photonSelect = ((photons.pt>20) & 
                        (abs(photons.eta) < 1.4442) &
                        (photons.isEE | photons.isEB) &
                        (photons.passEleVeto) & 
                        np.invert(photons.pixelSeed) & 
                        dRphomu & dRphoele
                       )
        
        
        #split out the ID requirement, enabling Iso and SIEIE to be inverted for control regions
        photonID = photons.photonId >= 2

        #parse VID cuts, define loose photons (not used yet)
        photon_MinPtCut = (photons.vidCuts>>0 & 3)>=2 
        photon_PhoSCEtaMultiRangeCut = (photons.vidCuts>>2 & 3)>=2 
        photon_PhoSingleTowerHadOverEmCut = (photons.vidCuts>>4 & 3)>=2  
        photon_PhoFull5x5SigmaIEtaIEtaCut = (photons.vidCuts>>6 & 3)>=2  
        photon_ChIsoCut = (photons.vidCuts>>8 & 3)>=2  
        photon_NeuIsoCut = (photons.vidCuts>>10 & 3)>=2  
        photon_PhoIsoCut = (photons.vidCuts>>12 & 3)>=2  
        
        photonID_NoChIsoSIEIE = (photon_MinPtCut & 
                                 photon_PhoSCEtaMultiRangeCut & 
                                 photon_PhoSingleTowerHadOverEmCut & 
                                 photon_PhoFull5x5SigmaIEtaIEtaCut & 
                                 photon_NeuIsoCut & 
                                 photon_PhoIsoCut)

        
        tightPhotons = photons[photonSelect & photonID]
        loosePhotons = photons[photonSelect & photonID_NoChIsoSIEIE & photon_PhoFull5x5SigmaIEtaIEtaCut]
        loosePhotonsSideband = photons[photonSelect & photonID_NoChIsoSIEIE & (photons.sieie>0.012)]
        
        ##medium jet ID cut
        jetIDbit = 1
        if year>2016: jetIDbit=2

        ##check dR jet,lepton & jet,photon
        jetMu = jets['p4'].cross(tightMuon['p4'],nested=True)
        dRjetmu = ((jetMu.i0.delta_r(jetMu.i1)).min()>0.4) | (tightMuon.counts==0)

        jetEle = jets['p4'].cross(tightElectron['p4'],nested=True)
        dRjetele = ((jetEle.i0.delta_r(jetEle.i1)).min()>0.4) | (tightElectron.counts==0)

        jetPho = jets['p4'].cross(tightPhotons['p4'],nested=True)
        dRjetpho = ((jetPho.i0.delta_r(jetPho.i1)).min()>0.1) | (tightPhotons.counts==0)
        
        jetSelect = ((jets.pt > 30) &
                     (abs(jets.eta) < 2.4) &
                     ((jets.jetId >> jetIDbit & 1)==1) &
                     dRjetmu & dRjetele & dRjetpho                    
                    )

        tightJets = jets[jetSelect]
        
        bTagWP = 0.6321   #2016 DeepCSV working point

        btagged = tightJets.btag>bTagWP

        bJets = tightJets[btagged]

        ## Define M3, mass of 3-jet pair with highest pT
        triJet = tightJets['p4'].choose(3)

        triJetPt = (triJet.i0 + triJet.i1 + triJet.i2).pt
        triJetMass = (triJet.i0 + triJet.i1 + triJet.i2).mass
        M3 = triJetMass[triJetPt.argmax()]


        leadingMuon = tightMuon[::1] 
        leadingElectron = tightElectron[::1]        
        
        leadingPhoton = tightPhotons[:,:1]
        leadingPhotonLoose = loosePhotons[:,:1]
        leadingPhotonSideband = loosePhotonsSideband[:,:1]

        
#        egammaMass = (leadingElectron['p4'] + leadingPhoton['p4']).mass
        egamma = leadingElectron['p4'].cross(leadingPhoton['p4'])
        mugamma = leadingMuon['p4'].cross(leadingPhoton['p4'])
        egammaMass = (egamma.i0 + egamma.i1).mass
        mugammaMass = (mugamma.i0 + mugamma.i1).mass
        
        
        
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
            isHadFake = ~(isMisIDele | isGenPho | isHadPho) & (leadingPhoton.counts==1)
            
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
            isHadFakeLoose = ~(isMisIDeleLoose | isGenPhoLoose | isHadPhoLoose) & (leadingPhotonLoose.counts==1)        

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
            isHadFakeSideband = ~(isMisIDeleSideband | isGenPhoSideband | isHadPhoSideband) & (leadingPhotonSideband.counts==1)        

            #define integer definition for the photon category axis
            phoCategorySideband = 1*isGenPhoSideband + 2*isMisIDeleSideband + 3*isHadPhoSideband + 4*isHadFakeSideband            
        else:
            phoCategory = np.ones_like(df['event'])
            phoCategoryLoose = np.ones_like(df['event'])
            phoCategorySideband = np.ones_like(df['event'])
        

        ### remove filter selection
        ###    This is already applied in the skim, and is causing data to fail for some reason (the flag branches are duplicated in NanoAOD for data, is it causing problems???)
#         mu_noLoose = (muTrigger & filters & passOverlapRemoval &
#                       oneMuon & eleVeto &
#                       looseMuonSel & looseElectronSel)
#         ele_noLoose = (eleTrigger & filters & passOverlapRemoval &
#                        oneEle & muVeto &
#                        looseMuonSel & looseElectronSel)

        mu_noLoose = (muTrigger & passOverlapRemoval &
                      oneMuon & eleVeto &
                      looseMuonSel & looseElectronSel)
        ele_noLoose = (eleTrigger & passOverlapRemoval &
                       oneEle & muVeto &
                       looseMuonSel & looseElectronSel)

        # lep_noLoose = mu_noLoose| ele_noLoose
        
        lep_jetSel = ((tightJets.counts >= 4) & (bJets.counts >= 1)
                     )
        lep_zeropho = (lep_jetSel & 
                       (tightPhotons.counts == 0)
                      )
        lep_phosel = (lep_jetSel & 
                      (tightPhotons.counts == 1)
                     )
        lep_phoselLoose = (lep_jetSel & 
                           (loosePhotons.counts == 1)
                          )
        lep_phoselSideband = (lep_jetSel & 
                              (loosePhotonsSideband.counts == 1)
                             )

        lep_phosel_3j0t = ((tightJets.counts >= 3) & (bJets.counts ==0) &
                           (tightPhotons.counts == 1)
                          )

#        lepFlavor = -0.5*ele_noLoose + 0.5*mu_noLoose
        
        
#        evtWeight = np.ones_like(df['event'],dtype=np.float64)        
        if not 'Data' in dataset:
            lumiWeight = np.ones_like(df['event'],dtype=np.float64)
            nMCevents = self.mcEventYields[datasetFull]
            xsec = crossSections[dataset]
            lumiWeight *= xsec * lumis[year] / nMCevents

#            evtWeight *= xsec * lumis[year] / nMCevents

            weights.add('lumiWeight',lumiWeight)

            #btag key name
            #name / working Point / type / systematic / jetType
            #  ... / 0-loose 1-medium 2-tight / comb,mujets,iterativefit / central,up,down / 0-b 1-c 2-udcsg 

            bJetSF_b = self.evaluator['btag%iDeepCSV_1_comb_central_0'%year](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btag)
            bJetSF_c = self.evaluator['btag%iDeepCSV_1_comb_central_1'%year](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btag)
            bJetSF_udcsg = self.evaluator['btag%iDeepCSV_1_incl_central_2'%year](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btag)

            bJetSF_b_up = self.evaluator['btag%iDeepCSV_1_comb_up_0'%year](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btag)
            bJetSF_c_up = self.evaluator['btag%iDeepCSV_1_comb_up_1'%year](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btag)
            bJetSF_udcsg_up = self.evaluator['btag%iDeepCSV_1_incl_up_2'%year](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btag)

            bJetSF_b_down = self.evaluator['btag%iDeepCSV_1_comb_down_0'%year](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btag)
            bJetSF_c_down = self.evaluator['btag%iDeepCSV_1_comb_down_1'%year](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btag)
            bJetSF_udcsg_down = self.evaluator['btag%iDeepCSV_1_incl_down_2'%year](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btag)

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
            pMC   = btagEfficiencies[btagged].prod()     * (1.-btagEfficiencies[np.invert(btagged)]).prod() 
            pData = btagEfficienciesData[btagged].prod() * (1.-btagEfficienciesData[np.invert(btagged)]).prod()
            pData_b_up = btagEfficienciesData_b_up[btagged].prod() * (1.-btagEfficienciesData_b_up[np.invert(btagged)]).prod()
            pData_b_down = btagEfficienciesData_b_down[btagged].prod() * (1.-btagEfficienciesData_b_down[np.invert(btagged)]).prod()
            pData_l_up = btagEfficienciesData_l_up[btagged].prod() * (1.-btagEfficienciesData_l_up[np.invert(btagged)]).prod()
            pData_l_down = btagEfficienciesData_l_down[btagged].prod() * (1.-btagEfficienciesData_l_down[np.invert(btagged)]).prod()

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
            
            eleSF = (eleID*eleRECO).prod()
            eleSF_up = ((eleID + eleIDerr) * (eleRECO + eleRECOerr)).prod()
            eleSF_down = ((eleID - eleIDerr) * (eleRECO - eleRECOerr)).prod()

            weights.add('eleEffWeight',weight=eleSF,weightUp=eleSF_up,weightDown=eleSF_down)

#            evtWeight *= eleSF

            muID = self.mu_id_sf(tightMuon.eta, tightMuon.pt)
            muIDerr = self.mu_id_err(tightMuon.eta, tightMuon.pt)
            muIso = self.mu_iso_sf(tightMuon.eta, tightMuon.pt)
            muIsoerr = self.mu_iso_err(tightMuon.eta, tightMuon.pt)
            muTrig = self.mu_iso_sf(abs(tightMuon.eta), tightMuon.pt)
            muTrigerr = self.mu_iso_err(abs(tightMuon.eta), tightMuon.pt)
            
            muSF = (muID*muIso*muTrig).prod()
            muSF_up = ((muID + muIDerr) * (muIso + muIsoerr) * (muTrig + muTrigerr)).prod()
            muSF_down = ((muID - muIDerr) * (muIso - muIsoerr) * (muTrig - muTrigerr)).prod()

            weights.add('muEffWeight',weight=muSF,weightUp=muSF_up, weightDown=muSF_down)

#            evtWeight *= muSF
        
        systList = ['noweight','nominal','muEffWeightUp','muEffWeightDown','eleEffWeightUp','eleEffWeightDown','btagWeight_lightUp','btagWeight_lightDown','btagWeight_heavyUp','btagWeight_heavyDown']

        if isData:
            systList = ['noweight']

        for syst in systList:
            
            weightSyst = syst
            if syst in ['nominal']:
                weightSyst=None

            if syst=='noweight':
                evtWeight = np.ones_like(df['event'])
            else:
                evtWeight = weights.weight(weightSyst)

            for lepton in ['electron','muon']:
                if lepton=='electron':
                    lepSel = ele_noLoose
                if lepton=='muon':
                    lepSel = mu_noLoose
    
                output['photon_pt'].fill(dataset=dataset,
                                         pt=tightPhotons.p4.pt[:,:1][lep_phosel & lepSel].flatten(),
                                         category=phoCategory[lep_phosel & lepSel].flatten(),
                                         lepFlavor=lepton,
                                         systematic=syst,
                                         weight=evtWeight[lep_phosel & lepSel].flatten())
    
                output['photon_eta'].fill(dataset=dataset,
                                          eta=tightPhotons.eta[:,:1][lep_phosel & lepSel].flatten(),
                                          category=phoCategory[lep_phosel & lepSel].flatten(),
                                          lepFlavor=lepton,
                                          systematic=syst,
                                          weight=evtWeight[lep_phosel & lepSel].flatten())
                
                output['photon_chIsoSideband'].fill(dataset=dataset,
                                                    chIso=loosePhotonsSideband.chIso[:,:1][lep_phoselSideband & lepSel].flatten(),
                                                    category=phoCategorySideband[lep_phoselSideband & lepSel].flatten(),
                                                    lepFlavor=lepton,
                                                    systematic=syst,
                                                    weight=evtWeight[lep_phoselSideband & lepSel].flatten())
                
                output['photon_chIso'].fill(dataset=dataset,
                                            chIso=loosePhotons.chIso[:,:1][lep_phoselLoose & lepSel].flatten(),
                                            category=phoCategoryLoose[lep_phoselLoose & lepSel].flatten(),
                                            lepFlavor=lepton,
                                            systematic=syst,
                                            weight=evtWeight[lep_phoselLoose & lepSel].flatten())
                
                
                output['M3'].fill(dataset=dataset,
                                  M3=M3[lep_phosel & lepSel].flatten(),
                                  category=phoCategoryLoose[lep_phosel & lepSel].flatten(),
                                  lepFlavor=lepton,
                                  systematic=syst,
                                  weight=evtWeight[lep_phosel & lepSel].flatten())
                
                output['M3Presel'].fill(dataset=dataset,
                                        M3=M3[lep_zeropho & lepSel].flatten(),
                                        lepFlavor=lepton,
                                        systematic=syst,
                                        weight=evtWeight[lep_zeropho & lepSel].flatten())                            
    
            
            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=egammaMass[lep_phosel & ele_noLoose].flatten(),
                                              category=phoCategory[lep_phosel & ele_noLoose].flatten(),
                                              lepFlavor='electron',
                                              systematic=syst,
                                              weight=evtWeight[lep_phosel & ele_noLoose].flatten())
            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=mugammaMass[lep_phosel & mu_noLoose].flatten(),
                                              category=phoCategory[lep_phosel & mu_noLoose].flatten(),
                                              lepFlavor='muon',
                                              systematic=syst,
                                              weight=evtWeight[lep_phosel & mu_noLoose].flatten())
    
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=egammaMass[lep_phosel_3j0t & ele_noLoose].flatten(),
                                                   category=phoCategory[lep_phosel_3j0t & ele_noLoose].flatten(),
                                                   lepFlavor='electron',
                                                   systematic=syst,
                                                   weight=evtWeight[lep_phosel_3j0t & ele_noLoose].flatten())
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=mugammaMass[lep_phosel_3j0t & mu_noLoose].flatten(),
                                                   category=phoCategory[lep_phosel_3j0t & mu_noLoose].flatten(),
                                                   lepFlavor='muon',
                                                   systematic=syst,
                                                   weight=evtWeight[lep_phosel_3j0t & mu_noLoose].flatten())
            

        output['EventCount'] = len(df['event'])

        return output

    def postprocess(self, accumulator):
        return accumulator



