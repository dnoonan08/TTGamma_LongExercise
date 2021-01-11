import time

from coffea import hist, util
import coffea.processor as processor
from coffea.nanoevents.methods import nanoaod
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools import extractor, dense_lookup
from coffea.btag_tools import BTagScaleFactor

import awkward as ak
import numpy as np
import pickle
import re

from .utils.crossSections import *
from .utils.genParentage import maxHistoryPDGID
from .utils.updateJets import updateJetP4

import os.path
cwd = os.path.dirname(__file__)


taggingEffLookup = util.load(f'{cwd}/utils/taggingEfficienciesDenseLookup.coffea')

bJetScales = BTagScaleFactor(f"{cwd}/ScaleFactors/Btag/DeepCSV_2016LegacySF_V1.btag.csv","MEDIUM")

puLookup = util.load(f'{cwd}/ScaleFactors/puLookup.coffea')
puLookup_Down = util.load(f'{cwd}/ScaleFactors/puLookup_Down.coffea')
puLookup_Up = util.load(f'{cwd}/ScaleFactors/puLookup_Up.coffea')


"""
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
"""


# Look at ProcessorABC to see the expected methods and what they are supposed to do
class TTGammaProcessor(processor.ProcessorABC):
#     def __init__(self, runNum = -1, eventNum = -1):
    def __init__(self, isMC=False, runNum=-1, eventNum=-1, mcEventYields=None, jetSyst='nominal'):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################
        ak.behavior.update(nanoaod.behavior)

        #self.mcEventYields = mcEventYields
        self.isMC = isMC

        if not jetSyst in ['nominal','JERUp','JERDown','JESUp','JESDown']:
            raise Exception(f'{jetSyst} is not in acceptable jet systematic types [nominal, JERUp, JERDown, JESUp, JESDown]')

        self.jetSyst = jetSyst

        dataset_axis = hist.Cat("dataset", "Dataset")
        lep_axis = hist.Cat("lepFlavor", "Lepton Flavor")

        systematic_axis = hist.Cat("systematic", "Systematic Uncertainty")

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
        
        ### Accumulator for holding histograms
        self._accumulator = processor.dict_accumulator({
            ##photon histograms
            'pho_pt'                 : hist.Hist("Counts", dataset_axis, pt_axis),
            'photon_pt'              : hist.Hist("Counts", dataset_axis, pt_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_eta'             : hist.Hist("Counts", dataset_axis, eta_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_chIso'           : hist.Hist("Counts", dataset_axis, chIso_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_lepton_mass'     : hist.Hist("Counts", dataset_axis, mass_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_lepton_mass_3j0t': hist.Hist("Counts", dataset_axis, mass_axis, phoCategory_axis, lep_axis, systematic_axis),
            'M3'                     : hist.Hist("Counts", dataset_axis, m3_axis, phoCategory_axis, lep_axis, systematic_axis),
            'EventCount'             : processor.value_accumulator(int),
        })

        self.ele_id_sf = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/ele_id_sf.coffea')
        self.ele_id_err = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/ele_id_err.coffea')

        self.ele_reco_sf = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/ele_reco_sf.coffea')
        self.ele_reco_err = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/ele_reco_err.coffea')


        self.mu_id_sf = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/mu_id_sf.coffea')
        self.mu_id_err = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/mu_id_err.coffea')

        self.mu_iso_sf = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/mu_iso_sf.coffea')
        self.mu_iso_err = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/mu_iso_err.coffea')

        self.mu_trig_sf = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/mu_trig_sf.coffea')
        self.mu_trig_err = util.load(f'{cwd}/ScaleFactors/MuEGammaScaleFactors/mu_trig_err.coffea')

        """
        ext = extractor()
        ext.add_weight_sets([f"btag2016 * {cwd}/ScaleFactors/Btag/DeepCSV_2016LegacySF_V1.btag.csv"])
        ext.finalize()
        self.evaluator = ext.make_evaluator()
        """
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        output['EventCount'] = len(events)

        datasetFull = events.metadata['dataset']
        dataset=datasetFull.replace('_2016','')
        #isData = 'Data' in dataset
        
        rho = events.fixedGridRhoFastjetAll

        #Temporary patch so we can add photon and lepton four vectors. Not needed for newer versions of NanoAOD
        events["Photon","charge"] = 0
        #Calculate charged hadron isolation for photons
        events["Photon","chIso"] = (events.Photon.pfRelIso03_chg)*(events.Photon.pt)

        #Calculate the maximum pdgID of any of the particles in the GenPart history
        if self.isMC:
            idx = ak.to_numpy(ak.flatten(abs(events.GenPart.pdgId)))
            par = ak.to_numpy(ak.flatten(events.GenPart.genPartIdxMother))
            num = ak.to_numpy(ak.num(events.GenPart.pdgId))        
            maxParentFlatten = maxHistoryPDGID(idx,par,num)
            events["GenPart","maxParent"] = ak.unflatten(maxParentFlatten, num)

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

            #potential overlap photons are only those passing the kinematic cuts 
            #if the overlap photon is actually from a non prompt decay (maxParent > 37), it's not part of the phase space of the separate sample 
            overlapPhoSelect = ((events.GenPart.pt>=overlapPt) & 
                                (abs(events.GenPart.eta) < overlapEta) & 
                                (events.GenPart.pdgId==22) & 
                                (events.GenPart.status==1) & 
                                (events.GenPart.maxParent < 37)
                               )
            overlapPhotons = events.GenPart[overlapPhoSelect] 

            #also require that photons are separate from all other gen particles
            #don't consider neutrinos and don't calculate the dR between the overlapPhoton and itself
            #ARH: do we need some tiny minimum pt requirement on these gen particles? (wasn't used in 2020)
            finalGen = events.GenPart[((events.GenPart.status==1)|(events.GenPart.status==71)) & (events.GenPart.pt > 0.01) &
                                      ~((abs(events.GenPart.pdgId)==12) | (abs(events.GenPart.pdgId)==14) | (abs(events.GenPart.pdgId)==16)) &
                                      ~overlapPhoSelect]

            #calculate dR between overlap photons and nearest gen particle
            phoGen, phoGenDR = overlapPhotons.nearest(finalGen, return_metric = True)
            phoGenMask = ak.fill_none(phoGenDR > overlapDR, True)

            #the event is overlapping with the separate sample if there is an overlap photon passing the dR cut, kinematic cuts, and not coming from hadronic activity
            isOverlap = ak.any(phoGenMask, axis=-1)
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
        # Remove photons that are within 0.4 of a lepton
        # phoMuDR is the delta R value to the nearest muon 
        # ak.fill_none is used to set the mask value to True when there are no muons in the event
        phoMu, phoMuDR  = events.Photon.nearest(tightMuon,return_metric=True)
        phoMuMask = ak.fill_none(phoMuDR > 0.4, True)
        
        phoEle, phoEleDR = events.Photon.nearest(tightElectron, return_metric=True)
        phoEleMask = ak.fill_none(phoEleDR > 0.4, True)

        #photon selection (no ID requirement used here)
        photonSelect = ((events.Photon.pt>20) & 
                        (abs(events.Photon.eta) < 1.4442) &
                        (events.Photon.isScEtaEE | events.Photon.isScEtaEB) &
                        (events.Photon.electronVeto) & 
                        np.invert(events.Photon.pixelSeed) & 
                        phoMuMask & phoEleMask
                       )
        
        #split out the ID requirement, enabling Iso to be inverted for control regions
        photonID = events.Photon.cutBased >= 2

        #parse VID cuts, define loose photons (not used yet)
        photon_MinPtCut = (events.Photon.vidNestedWPBitmap>>0 & 3)>=2 
        photon_PhoSCEtaMultiRangeCut = (events.Photon.vidNestedWPBitmap>>2 & 3)>=2 
        photon_PhoSingleTowerHadOverEmCut = (events.Photon.vidNestedWPBitmap>>4 & 3)>=2  
        photon_PhoFull5x5SigmaIEtaIEtaCut = (events.Photon.vidNestedWPBitmap>>6 & 3)>=2  
        photon_ChIsoCut = (events.Photon.vidNestedWPBitmap>>8 & 3)>=2  
        photon_NeuIsoCut = (events.Photon.vidNestedWPBitmap>>10 & 3)>=2  
        photon_PhoIsoCut = (events.Photon.vidNestedWPBitmap>>12 & 3)>=2  

        #photons passing all ID requirements, without the charged hadron isolation cut applied
        photonID_NoChIso = (photon_MinPtCut & 
                            photon_PhoSCEtaMultiRangeCut & 
                            photon_PhoSingleTowerHadOverEmCut & 
                            photon_PhoFull5x5SigmaIEtaIEtaCut & 
                            photon_NeuIsoCut & 
                            photon_PhoIsoCut)

        # 1. ADD SELECTION
        #  Object selection
        #select tightPhoton, the subset of photons passing the photonSelect cut and the photonID cut        
        tightPhoton = events.Photon[photonSelect & photonID]
        #select loosePhoton, the subset of photons passing the photonSelect cut and all photonID cuts without the charged hadron isolation cut applied
        loosePhoton = events.Photon[photonSelect & photonID_NoChIso]


        #ARH: ADD JET TRANSFORMER HERE!
        """
        #update jet kinematics based on jete energy systematic uncertainties
        if self.isMC:
        genJet = JaggedCandidateArray.candidatesfromcounts(
                df['nGenJet'],
                pt = df['GenJet_pt'],
                eta = df['GenJet_eta'],
                phi = df['GenJet_phi'],
                mass = df['GenJet_mass'],
            )
            jets.genJetIdx[jets.genJetIdx>=events.GenJet.counts] = -1 #fixes a bug in events.GenJet indices, skimmed after events.GenJet matching
            jets['ptGenJet'][jets.genJetIdx>-1] = events.GenJet[jets.genJetIdx[jets.genJetIdx>-1]].pt
            jets['rho'] = jets.pt.ones_like()*rho
            #adds additional columns to the jets array, containing the jet pt with JEC and JER variations
            #    additional columns added to jets:  pt_jer_up,   mass_jer_up
            #                                       pt_jer_down, mass_jer_down
            #                                       pt_jes_up,   mass_jes_up
            #                                       pt_jes_down, mass_jes_down
            Jet_transformer.transform(jets)
            # 4. ADD SYSTEMATICS
            #   If processing a jet systematic (based on value of self.jetSyst variable) update the jet pt and mass to reflect the jet systematic uncertainty variations
            #   Use the function updateJetP4(jets, pt=NEWPT, mass=NEWMASS) to update the pt and mass
        """
        

        ##check dR jet,lepton & jet,photon
        jetMu, jetMuDR = events.Jet.nearest(tightMuon, return_metric=True)
        jetMuMask = ak.fill_none(jetMuDR > 0.4, True)

        jetEle, jetEleDR = events.Jet.nearest(tightElectron, return_metric=True)
        jetEleMask = ak.fill_none(jetEleDR > 0.4, True)

        jetPho, jetPhoDR = events.Jet.nearest(tightPhoton, return_metric=True)
        jetPhoMask = ak.fill_none(jetPhoDR > 0.4, True)

        # 1. ADD SELECTION
        #select good jets
        # jets should have a pt of at least 30 GeV, |eta| < 2.4, pass the medium jet id (bit-wise selected from the jetID variable), and pass the delta R cuts defined above
        ##medium jet ID cut
        jetIDbit = 1

        jetSelectNoPt = ((abs(events.Jet.eta) < 2.4) &
                         ((events.Jet.jetId >> jetIDbit & 1)==1) &
                         jetMuMask & jetEleMask & jetPhoMask )
        
        jetSelect = jetSelectNoPt & (events.Jet.pt > 30)

        # 1. ADD SELECTION
        #select the subset of jets passing the jetSelect cuts
        tightJet = events.Jet[jetSelect]

        # 1. ADD SELECTION
        # select the subset of tightJet which pass the Deep CSV tagger
        bTagWP = 0.6321   #2016 DeepCSV working point
        btagged = tightJet.btagDeepB>bTagWP  
        bTaggedJet= tightJet[btagged]


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
        muon_eventSelection = (muTrigger & passOverlapRemoval & 
                               oneMuon & eleVeto & 
                               looseMuonVeto & looseElectronVeto) 

        # electron selection, requires events to pass:   electron trigger
        #                                                overlap removal
        #                                                have exactly one electron
        #                                                have no muons
        #                                                have no loose muons
        #                                                have no loose electrons
        electron_eventSelection = (eleTrigger & passOverlapRemoval &
                                   oneEle & muVeto & 
                                   looseMuonVeto & looseElectronVeto)  

        # 1. ADD SELECTION
        #add selection 'eleSel', for events passing the electron event selection, and muSel for those passing the muon event selection
        #  ex: selection.add('testSelection', ak.to_numpy(event_mask))
    
        #ARH CoffeaQ: is there a better way than ak.to_numpy?
        #create a selection object
        selection = processor.PackedSelection()

        selection.add('eleSel',ak.to_numpy(electron_eventSelection))
        selection.add('muSel',ak.to_numpy(muon_eventSelection))

        #add two jet selection criteria
        #   First, 'jetSel' which selects events with at least 4 tightJet and at least one bTaggedJet
        nJets = 4
        selection.add('jetSel',     ak.to_numpy( (ak.num(tightJet) >= nJets) & (ak.num(bTaggedJet) >= 1) ))
        #   Second, 'jetSel_3j0t' which selects events with at least 3 tightJet and exactly zero bTaggedJet
        selection.add('jetSel_3j0t', ak.to_numpy((ak.num(tightJet) >= 3)     & (ak.num(bTaggedJet) == 0) ))

        # add selection for events with exactly 0 tight photons
        selection.add('zeroPho', ak.to_numpy(ak.num(tightPhoton) == 0))

        # add selection for events with exactly 1 tight photon
        selection.add('onePho',  ak.to_numpy(ak.num(tightPhoton) == 1))

        # add selection for events with exactly 1 loose photon
        selection.add('loosePho',ak.to_numpy(ak.num(loosePhoton) == 1))

        ##################
        # EVENT VARIABLES
        ##################

        # PART 2A: Uncomment to begin implementing event variables
        # 2. DEFINE VARIABLES
        ## Define M3, mass of 3-jet pair with highest pT
        # find all possible combinations of 3 tight jets in the events 
        #hint: using the ak.combinations(array,n) method chooses n unique items from array. Use the "fields" option to define keys you can use to access the items
        triJet=ak.combinations(tightJet,3,fields=["first","second","third"])
        triJetPt = (triJet.first + triJet.second + triJet.third).pt
        triJetMass = (triJet.first + triJet.second + triJet.third).mass
        M3 = triJetMass[ak.argmax(triJetPt,axis=-1,keepdims=True)]

        leadingMuon = tightMuon[:,:1]
        leadingElectron = tightElectron[:,:1]

        leadingPhoton = tightPhoton[:,:1]
        leadingPhotonLoose = loosePhoton[:,:1]

        # 2. DEFINE VARIABLES
        # define egammaMass, mass of combinations of tightElectron and leadingPhoton (hint: using the ak.cartesian() method)
        egammaPairs = ak.cartesian({"pho":leadingPhoton, "ele": tightElectron})
        # avoid erros when egammaPairs is empty
        if ak.all(ak.num(egammaPairs)==0):
            egammaMass =  np.ones((len(events),1))*-1
        else:
            egammaMass = (egammaPairs.pho + egammaPairs.ele).mass
        # define mugammaMass, mass of combinations of tightMuon and leadingPhoton (hint: using the ak.cartesian() method) 
        mugammaPairs = ak.cartesian({"pho":leadingPhoton, "mu":tightMuon})
        if ak.all(ak.num(mugammaPairs)==0):
            mugammaMass = np.ones((len(events),1))*-1
        else:
            mugammaMass = (mugammaPairs.pho + mugammaPairs.mu).mass

        ###################
        # PHOTON CATEGORIES
        ###################

        # Define photon category for each event

        phoCategory = np.ones(len(events))
        phoCategoryLoose = np.ones(len(events))

        # PART 2B: Uncomment to begin implementing photon categorization
        if self.isMC:
            #### Photon categories, using pdgID of the matched gen particle for the leading photon in the event
            # reco photons matched to a generated photon
            matchedPho = ak.any(leadingPhoton.matched_gen.pdgId==22, axis=-1)
            # reco photons really generated as electrons
            matchedEle =  ak.any(abs(leadingPhoton.matched_gen.pdgId)==11, axis=-1)
            # if the gen photon has a PDG ID > 25 in it's history, it has a hadronic parent
            hadronicParent = ak.any(leadingPhoton.matched_gen.maxParent>25, axis=-1)
            
            #####
            # 2. DEFINE VARIABLES
            # define the photon categories for tight photon events
            # a genuine photon is a reconstructed photon which is matched to a generator level photon, and does not have a hadronic parent
            isGenPho = matchedPho & ~hadronicParent
            # a hadronic photon is a reconstructed photon which is matched to a generator level photon, but has a hadronic parent
            isHadPho = matchedPho & hadronicParent
            # a misidentified electron is a reconstructed photon which is matched to a generator level electron
            isMisIDele = matchedEle
            # a hadronic/fake photon is a reconstructed photon that does not fall within any of the above categories
            isHadFake = ~(isMisIDele | isGenPho | isHadPho) & (ak.num(leadingPhoton)==1)

            #define integer definition for the photon category axis 
            phoCategory = 1*isGenPho + 2*isMisIDele + 3*isHadPho + 4*isHadFake

        
            # do photon matching for loose photons as well
            # reco photons matched to a generated photon 
            matchedPhoLoose = ak.any(leadingPhotonLoose.matched_gen.pdgId==22, axis=-1)
            # reco photons really generated as electrons 
            matchedEleLoose =  ak.any(abs(leadingPhotonLoose.matched_gen.pdgId)==11, axis=-1)
            # if the gen photon has a PDG ID > 25 in it's history, it has a hadronic parent
            hadronicParentLoose = ak.any(leadingPhotonLoose.matched_gen.maxParent>25, axis=-1)

            #####
            # 2. DEFINE VARIABLES
            # a genuine photon is a reconstructed photon which is matched to a generator level photon, and does not have a hadronic parent
            isGenPhoLoose = matchedPhoLoose & ~hadronicParentLoose
            # a hadronic photon is a reconstructed photon which is matched to a generator level photon, but has a hadronic parent
            isHadPhoLoose = matchedPhoLoose & hadronicParentLoose
            # a misidentified electron is a reconstructed photon which is matched to a generator level electron
            isMisIDeleLoose = matchedEle
            # a hadronic/fake photon is a reconstructed photon that does not fall within any of the above categories
            isHadFakeLoose = ~(isMisIDeleLoose | isGenPhoLoose | isHadPhoLoose) & (ak.num(leadingPhotonLoose)==1)        

            #define integer definition for the photon category axis
            phoCategoryLoose = 1*isGenPhoLoose + 2*isMisIDeleLoose + 3*isHadPhoLoose + 4*isHadFakeLoose
        
        ################
        # EVENT WEIGHTS
        ################

        #create a processor Weights object, with the same length as the number of events in the chunk
        weights = processor.Weights(len(events))

        if self.isMC:
            lumiWeight = np.ones(len(events))
            nMCevents = self.mcEventYields[datasetFull]
            xsec = crossSections[dataset]
            luminosity = 35860.0
            lumiWeight *= xsec * luminosity / nMCevents 

            #weights.add('lumiWeight',lumiWeight)

            puWeight = puLookup[datasetFull](events.Pileup.nTrueInt)
            puWeight_Up = puLookup_Up[datasetFull](events.Pileup.nTrueInt)
            puWeight_Down = puLookup_Down[datasetFull](events.Pileup.nTrueInt)
            
            weights.add('puWeight',weight=puWeight, weightUp=puWeight_Up, weightDown=puWeight_Down)

            #btag key name
            #name / working Point / type / systematic / jetType
            #  ... / 0-loose 1-medium 2-tight / comb,mujets,iterativefit / central,up,down / 0-b 1-c 2-udcsg 

            bJetSF = bJetScales('central',tightJet.hadronFlavour, abs(tightJet.eta), tightJet.pt)
            bJetSF_up = bJetScales('up',tightJet.hadronFlavour, abs(tightJet.eta), tightJet.pt)
            bJetSF_down = bJetScales('down',tightJet.hadronFlavour, abs(tightJet.eta), tightJet.pt)

            ## mc efficiency lookup, data efficiency is eff* scale factor
            taggingName = "TTGamma_SingleLept_2016"
            if datasetFull in taggingEffLookup:
                taggingName = datasetFull
            btagEfficiencies = taggingEffLookup[taggingName](tightJet.hadronFlavour,tightJet.pt,abs(tightJet.eta))
            btagEfficienciesData = btagEfficiencies*bJetSF
            btagEfficienciesData_up   = btagEfficiencies*bJetSF_up
            btagEfficienciesData_down = btagEfficiencies*bJetSF_down

            ##probability is the product of all efficiencies of tagged jets, times product of 1-eff for all untagged jets
            ## https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
            pMC          = ak.prod(btagEfficiencies[btagged], axis=-1)           * ak.prod((1.-btagEfficiencies[np.invert(btagged)]), axis=-1) 
            pData        = ak.prod(btagEfficienciesData[btagged], axis=-1)       * ak.prod((1.-btagEfficienciesData[np.invert(btagged)]),axis=-1)
            pData_up   = ak.prod(btagEfficienciesData_up[btagged], axis=-1)  * ak.prod((1.-btagEfficienciesData_up[np.invert(btagged)]),axis=-1)
            pData_down = ak.prod(btagEfficienciesData_down[btagged],axis=-1) * ak.prod((1.-btagEfficienciesData_down[np.invert(btagged)]),axis=-1)

            pMC = ak.where(pMC==0,1,pMC)
            btagWeight = pData/pMC
            btagWeight_up = pData_up/pMC
            btagWeight_down = pData_down/pMC
  
            weights.add('btagWeight',weight=btagWeight, weightUp=btagWeight_up, weightDown=btagWeight_down)

            eleID = self.ele_id_sf(tightElectron.eta, tightElectron.pt)
            eleIDerr = self.ele_id_err(tightElectron.eta, tightElectron.pt)
            eleRECO = self.ele_reco_sf(tightElectron.eta, tightElectron.pt)
            eleRECOerr = self.ele_reco_err(tightElectron.eta, tightElectron.pt)
            
            eleSF = ak.prod((eleID*eleRECO), axis=-1)
            eleSF_up   = ak.prod(((eleID + eleIDerr) * (eleRECO + eleRECOerr)), axis=-1)
            eleSF_down = ak.prod(((eleID - eleIDerr) * (eleRECO - eleRECOerr)), axis=-1)

            # 4. SYSTEMATICS
            # add electron efficiency weights to the weight container
            weights.add('eleEffWeight',weight=eleSF,weightUp=eleSF_up,weightDown=eleSF_down)

        
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


            #in some samples, generator systematics are not available, in those case the systematic weights of 1. are used
            if ak.mean(ak.num(events.PSWeight))==1:
                weights.add('ISR',    weight=np.ones(len(events)),weightUp=np.ones(len(events)),weightDown=np.ones(len(events)))
                weights.add('FSR',    weight=np.ones(len(events)),weightUp=np.ones(len(events)),weightDown=np.ones(len(events)))
                weights.add('PDF',    weight=np.ones(len(events)),weightUp=np.ones(len(events)),weightDown=np.ones(len(events)))
                weights.add('Q2Scale',weight=np.ones(len(events)),weightUp=np.ones(len(events)),weightDown=np.ones(len(events)))

            #Otherwise, calculate the weights and systematic variations
            else:
                #PDF Uncertainty weights
                #avoid errors from 0/0 division
                LHEPdfWeight_0 = ak.where(events.LHEPdfWeight[:,0]==0,1,events.LHEPdfWeight[:,0])
                LHEPdfVariation = events.LHEPdfWeight / LHEPdfWeight_0
                weights.add('PDF', weight=np.ones(len(events)), 
                            weightUp=ak.max(LHEPdfVariation,axis=1), 
                            weightDown=ak.min(LHEPdfVariation,axis=1))

                #Q2 Uncertainty weights
                if ak.mean(ak.num(events.LHEScaleWeight)) == 9:
                    scaleWeightSelector=[0,1,3,5,7,8]
                elif ak.mean(ak.num(events.LHEScaleWeight)) == 44:
                    scaleWeightSelector=[0,5,15,24,34,39]
                else:
                    scaleWeightSelector=[]
                LHEScaleVariation = events.LHEScaleWeight[:,scaleWeightSelector]
                weights.add('Q2Scale', weight=np.ones(len(events)), 
                            weightUp=ak.max(LHEScaleVariation,axis=1), 
                            weightDown=ak.min(LHEScaleVariation,axis=1))

                #ISR / FSR uncertainty weights
                if not ak.all(events.Generator.weight==events.LHEWeight.originalXWGTUP):
                    psWeights = events.PSWeight * events.LHEWeight.originalXWGTUP / events.Generator.weight
                else:
                    psWeights = events.PSWeight

                weights.add('ISR',weight=np.ones(len(events)), weightUp=psWeights[:,2], weightDown=psWeights[:,0])
                weights.add('FSR',weight=np.ones(len(events)), weightUp=psWeights[:,3], weightDown=psWeights[:,1])
                

        ###################
        # FILL HISTOGRAMS
        ###################
        # PART 3: Uncomment to add histograms
#        systList = ['noweight','nominal']

        systList = ['nominal','muEffWeightUp','muEffWeightDown','eleEffWeightUp','eleEffWeightDown','ISRUp', 'ISRDown', 'FSRUp', 'FSRDown', 'PDFUp', 'PDFDown', 'Q2ScaleUp', 'Q2ScaleDown','puWeightUp','puWeightDown','btagWeightUp','btagWeightDown']
        #ARH: Missing: PU, btag

        # PART 4: SYSTEMATICS
        # uncomment the full list after systematics have been implemented        
        #systList = ['noweight','nominal','puWeightUp','puWeightDown','muEffWeightUp','muEffWeightDown','eleEffWeightUp','eleEffWeightDown','btagWeightUp','btagWeightDown','ISRUp', 'ISRDown', 'FSRUp', 'FSRDown', 'PDFUp', 'PDFDown', 'Q2ScaleUp', 'Q2ScaleDown']

        if not self.jetSyst=='nominal':
            systList=[self.jetSyst]

        if not self.isMC:
            systList = ['noweight']

        #ARH: temp hist for testing purposes
        output['pho_pt'].fill(dataset=dataset,
                              pt = ak.flatten(ak.fill_none(M3,-1)))
#                                 pt=ak.flatten(tightPhoton.pt[:,:1]))

        
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

            #loop over both electron and muon selections
            for lepton in ['electron','muon']:
                if lepton=='electron':
                    lepSel='eleSel'
                if lepton=='muon':
                    lepSel='muSel'

                # 3. GET HISTOGRAM EVENT SELECTION
                #  use the selection.all() method to select events passing 
                #  the lepton selection, 4-jet 1-tag jet selection, and either the one-photon or loose-photon selections
                #  ex: selection.all( *('LIST', 'OF', 'SELECTION', 'CUTS') )
                phosel = selection.all(*(lepSel, "jetSel", 'onePho'))
                phoselLoose = selection.all(*(lepSel, "jetSel", 'loosePho') )

                # 3. FILL HISTOGRAMS
                #    fill photon_pt and photon_eta, using the tightPhotons array, from events passing the phosel selection

                output['photon_pt'].fill(dataset=dataset,
                                         pt=ak.flatten(tightPhoton.pt[:,:1][phosel]),
                                         category=phoCategory[phosel],
                                         lepFlavor=lepton,
                                         systematic=syst,
                                         weight=evtWeight[phosel])
    
                output['photon_eta'].fill(dataset=dataset,
                                          eta=ak.flatten(tightPhoton.eta[:,:1][phosel]),
                                          category=phoCategory[phosel],
                                          lepFlavor=lepton,
                                          systematic=syst,
                                          weight=evtWeight[phosel])
                
                #    fill photon_chIso histogram, using the loosePhotons array (photons passing all cuts, except the charged hadron isolation cuts)
                output['photon_chIso'].fill(dataset=dataset,
                                            chIso=ak.flatten(loosePhoton.chIso[:,:1][phoselLoose]),
                                            category=phoCategoryLoose[phoselLoose],
                                            lepFlavor=lepton,
                                            systematic=syst,
                                            weight=evtWeight[phoselLoose])
                
                #    fill M3 histogram, for events passing the phosel selection
                output['M3'].fill(dataset=dataset,
                                  M3=ak.flatten(ak.fill_none(M3[phosel],-1)),
                                  category=phoCategory[phosel],
                                  lepFlavor=lepton,
                                  systematic=syst,
                                  weight=evtWeight[phosel])
                
            phosel_e = selection.all(*('eleSel', "jetSel", 'onePho') )
            phosel_mu = selection.all(*('muSel', "jetSel", 'onePho') )
                    
            # 3. GET HISTOGRAM EVENT SELECTION
            #  use the selection.all() method to select events passing the eleSel or muSel selection, 
            # and the 3-jet 0-btag selection, and have exactly one photon
      
            phosel_e = selection.all(*('eleSel', "jetSel", 'onePho') )
            phosel_mu = selection.all(*('muSel', "jetSel", 'onePho') )

            phosel_3j0t_e = selection.all(*('eleSel', "jetSel_3j0t", 'onePho') )
            phosel_3j0t_mu = selection.all(*('muSel', "jetSel_3j0t", 'onePho') )

            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=ak.flatten(egammaMass[phosel_e]),
                                              category=phoCategory[phosel_e],
                                              lepFlavor='electron',
                                              systematic=syst,
                                              weight=evtWeight[phosel_e])
                                        
            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=ak.flatten(mugammaMass[phosel_mu]),
                                              category=phoCategory[phosel_mu],
                                              lepFlavor='muon',
                                              systematic=syst,
                                              weight=evtWeight[phosel_mu])
    
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=ak.flatten(egammaMass[phosel_3j0t_e]),
                                                   category=phoCategory[phosel_3j0t_e],
                                                   lepFlavor='electron',
                                                   systematic=syst,
                                                   weight=evtWeight[phosel_3j0t_e])

            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=ak.flatten(mugammaMass[phosel_3j0t_mu]),
                                                   category=phoCategory[phosel_3j0t_mu],
                                                   lepFlavor='muon',
                                                   systematic=syst,
                                                   weight=evtWeight[phosel_3j0t_mu])
        
        return output

    def postprocess(self, accumulator):
        return accumulator
