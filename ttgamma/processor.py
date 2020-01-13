import time

from coffea import hist, util
from coffea.analysis_objects import JaggedCandidateArray, JaggedTLorentzVectorArray
import coffea.processor as processor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty, JetTransformer, JetResolution, JetResolutionScaleFactor
from coffea.lookup_tools import extractor, dense_lookup

import uproot

from awkward import JaggedArray
from uproot_methods.classes.TLorentzVector import TLorentzVectorArray
import numpy as np
import pickle

import re

from .utils.crossSections import *
from .utils.efficiencies import getMuSF, getEleSF

from .utils.genParentage import maxHistoryPDGID
from .utils.updateJets import updateJetP4

import os.path
cwd = os.path.dirname(__file__)

#load lookup tool for btagging efficiencies
with open(f'{cwd}/utils/taggingEfficienciesDenseLookup.pkl', 'rb') as _file:
    taggingEffLookup = pickle.load(_file)

#load lookup tools for pileup scale factors
puLookup = util.load(f'{cwd}/ScaleFactors/puLookup.coffea')
puLookup_Down = util.load(f'{cwd}/ScaleFactors/puLookup_Down.coffea')
puLookup_Up = util.load(f'{cwd}/ScaleFactors/puLookup_Up.coffea')


#create and load jet extractor
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

#list of JEC and JER correction names
jec_names = ['Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs','Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs']
junc_names = ['Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs']

jer_names = ['Summer16_25nsV1_MC_PtResolution_AK4PFchs']
jersf_names = ['Summer16_25nsV1_MC_SF_AK4PFchs']

#create JEC and JER correctors
JECcorrector = FactorizedJetCorrector(**{name: Jetevaluator[name] for name in jec_names})
JECuncertainties = JetCorrectionUncertainty(**{name:Jetevaluator[name] for name in junc_names})

JER = JetResolution(**{name:Jetevaluator[name] for name in jer_names})
JERsf = JetResolutionScaleFactor(**{name:Jetevaluator[name] for name in jersf_names})

Jet_transformer = JetTransformer(jec=JECcorrector,junc=JECuncertainties, jer = JER, jersf = JERsf)



# Look at ProcessorABC to see the expected methods and what they are supposed to do
class TTGammaProcessor(processor.ProcessorABC):
    def __init__(self, mcEventYields = None, jetSyst='nominal'):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################

        self.mcEventYields = mcEventYields

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
            # 3. ADD HISTOGRAMS
            ## book histograms for photon pt, eta, and charged hadron isolation
            #'photon_pt':
            #'photon_eta':
            #'photon_chIso':

            ## book histogram for photon/lepton mass in a 3j0t region
            #'photon_lepton_mass_3j0t':

            ## book histogram for M3 variable
            #'M3':

            'EventCount':processor.value_accumulator(int)
        })

        ext = extractor()
        ext.add_weight_sets([f"btag2016 * {cwd}/ScaleFactors/Btag/DeepCSV_2016LegacySF_V1.btag.csv"])
        ext.finalize()
        self.evaluator = ext.make_evaluator()
        
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
        
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()

        datasetFull = df['dataset']
        dataset=datasetFull.replace('_2016','')

        isData = 'Data' in dataset

        ################################
        # DEFINE JAGGED CANDIDATE ARRAYS
        ################################

        #load muon objects
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

        #load electron objects
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

        #load jet object
        jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'],
            eta=df['Jet_eta'],
            phi=df['Jet_phi'],
            mass=df['Jet_mass'],
            jetId=df['Jet_jetId'],
            btag=df['Jet_btagDeepB'],
            area=df['Jet_area'],
            ptRaw=df['Jet_pt'] * (1-df['Jet_rawFactor']),
            massRaw=df['Jet_mass'] * (1-df['Jet_rawFactor']),
            hadFlav=df['Jet_hadronFlavour'] if not isData else np.ones_like(df['Jet_jetId']),
            genJetIdx=df['Jet_genJetIdx'] if not isData else np.ones_like(df['Jet_jetId']),
            ptGenJet=np.zeros_like(df['Jet_pt']),
        )

        #load photon objects
        photons = JaggedCandidateArray.candidatesfromcounts(
            df['nPhoton'],
            pt=df['Photon_pt'],
            eta=df['Photon_eta'],
            phi=df['Photon_phi'],
            mass=np.zeros_like(df['Photon_pt']),
            isEE=df['Photon_isScEtaEE'],
            isEB=df['Photon_isScEtaEB'],
            photonId=df['Photon_cutBased'],
            passEleVeto=df['Photon_electronVeto'],
            pixelSeed=df['Photon_pixelSeed'],
            sieie=df['Photon_sieie'],
            chIso=df['Photon_pfRelIso03_chg']*df['Photon_pt'],
            vidCuts=df['Photon_vidNestedWPBitmap'],
            genFlav=df['Photon_genPartFlav'] if not isData else np.ones_like(df['Photon_electronVeto']),
            genIdx=df['Photon_genPartIdx'] if not isData else np.ones_like(df['Photon_electronVeto']),
        )

        rho = df['fixedGridRhoFastjetAll']
        
        if not isData:

            #load gen parton objects
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
            overlapPhoSelect = ((genPart.pt>=overlapPt) & 
                                (abs(genPart.eta) < overlapEta) & 
                                (genPart.pdgid==22) & 
                                (genPart.status==1)
                               )
            #potential overlap photons are only those passing the kinematic cuts
            OverlapPhotons = genPart[overlapPhoSelect] 

            #if the overlap photon is actually from a non prompt decay, it's not part of the phase space of the separate sample
            idx = OverlapPhotons.motherIdx
            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            
            finalGen = genPart[((genPart.status==1)|(genPart.status==71)) & ~((abs(genPart.pdgid)==12) | (abs(genPart.pdgid)==14) | (abs(genPart.pdgid)==16))]
            genPairs = OverlapPhotons['p4'].cross(finalGen['p4'],nested=True)
            ##remove the case where the cross produce is the gen photon with itself
            genPairs = genPairs[~(genPairs.i0==genPairs.i1)]
            #find closest gen particle to overlap photons
            dRPairs = genPairs.i0.delta_r(genPairs.i1)

            #the event is overlapping with the separate sample if there is an overlap photon passing the dR cut and not coming from hadronic activity
            isOverlap = ((dRPairs.min()>overlapDR) & (maxParent<37)).any()
            passOverlapRemoval = ~isOverlap
        else:
            passOverlapRemoval = np.ones_like(df['event'])==1
            


        ##################
        # OBJECT SELECTION
        ##################
        # PART 1A Uncomment to add in object selection
        """
        # 1. ADD SELECTION
        #select tight muons
        # tight muons should have a pt of at least 30 GeV, |eta| < 2.4, pass the tight muon ID cut (tightID variable), and have a relative isolation of less than 0.15
        muonSelectTight = ((?) &
                           (?) &
                           ? &
                           (?))

        #select loose muons
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


        #select tight electrons
        # 1. ADD SELECTION
        #select tight electrons
        # tight electrons should have a pt of at least 35 GeV, |eta| < 2.1, pass the cut based electron id (cutBased variable in NanoAOD>=4), and pass the etaGap, D0, and DZ cuts defined above
        electronSelectTight = ((?) &
                               (?) &
                               (?) &
                               ? & 
                               ? & 
                               ? )

        #select loose electrons
        electronSelectLoose = ((electrons.pt>15) & 
                               (abs(electrons.eta)<2.4) & 
                               (electrons.cutBased>=1) &
                               eleEtaGap &      
                               elePassD0 & 
                               elePassDZ & 
                               np.invert(electronSelectTight)
                              )
        
        # 1. ADD SELECTION
        #  Object selection
        #select the subset of muons passing the muonSelectTight and muonSelectLoose cuts
        tightMuon = ?
        looseMuon = ?
        
        # 1. ADD SELECTION
        #  Object selection
        #select the subset of electrons passing the electronSelectTight and electronSelectLoose cuts
        tightElectron = ?
        looseElectron = ?



        
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


        #parse VID cuts, define loose photons (photons without chIso cut)
        photon_MinPtCut = (photons.vidCuts>>0 & 3)>=2 
        photon_PhoSCEtaMultiRangeCut = (photons.vidCuts>>2 & 3)>=2 
        photon_PhoSingleTowerHadOverEmCut = (photons.vidCuts>>4 & 3)>=2  
        photon_PhoFull5x5SigmaIEtaIEtaCut = (photons.vidCuts>>6 & 3)>=2  
        photon_ChIsoCut = (photons.vidCuts>>8 & 3)>=2  
        photon_NeuIsoCut = (photons.vidCuts>>10 & 3)>=2  
        photon_PhoIsoCut = (photons.vidCuts>>12 & 3)>=2  
        
        #photons passing all ID requirements, without the charged hadron isolation cut applied
        photonID_NoChIso = (photon_MinPtCut & 
                            photon_PhoSCEtaMultiRangeCut & 
                            photon_PhoSingleTowerHadOverEmCut & 
                            photon_PhoFull5x5SigmaIEtaIEtaCut & 
                            photon_NeuIsoCut & 
                            photon_PhoIsoCut)
        
        # 1. ADD SELECTION
        #  Object selection
        #select tightPhotons, the subset of photons passing the photonSelect cut and the photonID cut
        tightPhotons = ?
        #select loosePhotons, the subset of photons passing the photonSelect cut and all photonID cuts without the charged hadron isolation cut applied
        loosePhotons = ?
        




        #update jet kinematics based on jete energy systematic uncertainties
        if not isData:
            genJet = JaggedCandidateArray.candidatesfromcounts(
                df['nGenJet'],
                pt = df['GenJet_pt'],
                eta = df['GenJet_eta'],
                phi = df['GenJet_phi'],
                mass = df['GenJet_mass'],
            )

            jets.genJetIdx[jets.genJetIdx>=genJet.counts] = -1 #fixes a but in genJet indices, skimmed after genJet matching

            jets['ptGenJet'][jets.genJetIdx>-1] = genJet[jets.genJetIdx[jets.genJetIdx>-1]].pt
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


        ##check dR jet,lepton & jet,photon
        jetMu = jets['p4'].cross(tightMuon['p4'],nested=True)
        dRjetmu = ((jetMu.i0.delta_r(jetMu.i1)).min()>0.4) | (tightMuon.counts==0)

        jetEle = jets['p4'].cross(tightElectron['p4'],nested=True)
        dRjetele = ((jetEle.i0.delta_r(jetEle.i1)).min()>0.4) | (tightElectron.counts==0)

        jetPho = jets['p4'].cross(tightPhotons['p4'],nested=True)
        dRjetpho = ((jetPho.i0.delta_r(jetPho.i1)).min()>0.1) | (tightPhotons.counts==0)



        # 1. ADD SELECTION
        #select good jets
        # jetsshould have a pt of at least 30 GeV, |eta| < 2.4, pass the medium jet id (bit-wise selected from the jetID variable), and pass the delta R cuts defined above (dRjetmu, dRjetele, dRjetpho)
        jetSelect = ((?) &
                     (?) &
                     ((jets.jetId >> 1 & 1)==1) &
                     ? & ? & ? )
        
        # 1. ADD SELECTION
        #select the subset of jets passing the jetSelect cuts
        tightJets = ?


        #find jets passing DeepCSV medium working point
        bTagWP = 0.6321   #2016 DeepCSV working point

        # 1. ADD SELECTION
        # select the subset of tightJets which pass the Deep CSV tagger
        bTaggedJets = ?
        """


        #####################
        # EVENT SELECTION
        #####################
        ### PART 1B: Uncomment to add event selection
        """
        # 1. ADD SELECTION
        ## apply triggers
        # muon events should be triggered by either the HLT_IsoMu24 or HLT_IsoTkMu24 triggers
        # electron events should be triggered by HLT_Ele27_WPTight_Gsf trigger
        # HINT: trigger values can be accessed with the variable df['TRIGGERNAME'], 
        # the bitwise or operator can be used to select multiple triggers df['TRIGGER1'] | df['TRIGGER2']
        muTrigger = ?
        eleTrigger = ?

        # 1. ADD SELECTION
        #  Event selection
        #oneMuon, should be true if there is exactly one tight muon in the event (hint, the .counts method returns the number of objects in each row of a jagged array)
        oneMuon = ?
        #muVeto, should be true if there are no tight muons in the event
        muVeto = ?

        # 1. ADD SELECTION
        #  Event selection
        #oneEle should be true if there is exactly one tight electron in the event
        oneEle = ?
        #eleVeto should be true if there are no tight electrons in the event
        eleVeto = ?

        # 1. ADD SELECTION
        #  Event selection
        #looseMuonSel and looseElectronSel should be tru if there are 0 loose muons or electrons in the event
        looseMuonSel = ?
        looseElectronSel = ?


        # 1. ADD SELECTION
        # muon selection, requires events to pass:   muon trigger
        #                                            overlap removal
        #                                            have exactly one muon
        #                                            have no electrons
        #                                            have no loose muons
        #                                            have no loose electrons
        muon_eventSelection = ?
        # electron selection, requires events to pass:   electron trigger
        #                                                overlap removal
        #                                                have exactly one electron
        #                                                have no muons
        #                                                have no loose muons
        #                                                have no loose electrons
        electron_eventSelection = ?

        #create a selection object
        selection = processor.PackedSelection()

        # 1. ADD SELECTION
        #add selection 'eleSel', for events passing the electron event selection, and muSel for those passing the muon event selection
        #  ex: selection.add('testSelection', array_of_booleans)
        selection.add('eleSel', ???)
        selection.add('muSel', ???)

        #add two jet selection criteria
        #   First, 'jetSel' which selects events with at least 4 tightJets and at least one bTaggedJets
        selection.add('jetSel', ???)
        #   Second, 'jetSel_3j0t' which selects events with at least 3 tightJets and exactly zero bTaggedJets
        selection.add('jetSel_3j0t', ???)

        # add selection for events with exactly 0 tight photons
        selection.add('zeroPho', ?)
        # add selection for events with exactly 1 tight photon
        selection.add('onePho', ?)
        # add selection for events with exactly 1 loose photon
        selection.add('loosePho', ?)
        """

        ##################
        # EVENT VARIABLES
        ##################

        # PART 2A: Uncomment to begin implementing event variables
        """        
        # 2. DEFINE VARIABLES
        ## Define M3, mass of 3-jet pair with highest pT
        # find all possible combinations of 3 tight jets in the events (hint: using the .p4.choose() method of jagged arrays to do combinations of the TLorentzVectors) 
        triJet = ?
        # calculate
        triJetPt = ?
        triJetMass = ?
        # define the M3 variable, the triJetMass of the combination with the highest triJetPt value (hint: using the .argmax() method)
        M3 = ?

        leadingPhoton = tightPhotons[:,:1]
        leadingPhotonLoose = loosePhotons[:,:1]

        # 2. DEFINE VARIABLES
        # define egammaMass, mass of combinations of tightElectron and leadingPhoton (hint: using the .cross() method)
        egammaPairs = ?
        egammaMass = ?
        # define egammaMass, mass of combinations of tightElectron and leadingPhoton (hint: using the .cross() method)
        mugammaPairs = ?
        mugammaMass = ?
        """
        
        ###################
        # PHOTON CATEGORIES
        ###################

        # Define photon category for each event

        phoCategory = np.ones(df.size)
        phoCategoryLoose = np.ones(df.size)

        # PART 2B: Uncomment to begin implementing photon categorization
        """
        if not isData:
            #### Photon categories, using genIdx branch of the leading photon in the event
            idx = leadingPhoton.genIdx
            
            # look through gen particle history, finding the highest PDG ID
            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            # reco photons matched to a generated photon
            matchedPho = (genpdgid[idx]==22).any()
            # reco photons really generated as electrons
            matchedEle = (abs(genpdgid[idx])==11).any()

            # if the gen photon has a PDG ID > 25 in it's history, it has a hadronic parent
            hadronicParent = maxParent>25


            #####
            # 2. DEFINE VARIABLES
            # define the photon categories for tight photon events
            # a genuine photon is a reconstructed photon which is matched to a generator level photon, and does not have a hadronic parent
            isGenPho = ?
            # a hadronic photon is a reconstructed photon which is matched to a generator level photon, but has a hadronic parent
            isHadPho = ?
            # a misidentified electron is a reconstructed photon which is 
            isMisIDele = ?
            # a hadronic/fake photon is a reconstructed photon that does not fall within any of the above categories
            isHadFake = ?
            
            #define integer definition for the photon category axis
            phoCategory = 1*isGenPho + 2*isMisIDele + 3*isHadPho + 4*isHadFake
            

            # do photon matching for loose photons as well
            # look through parentage to find if any hadrons in genPhoton parent history
            idx = leadingPhotonLoose.genIdx

            # reco photons matched to a generated photon
            matchedPhoLoose = (genpdgid[idx]==22).any()
            # reco photons really generated as electrons
            matchedEleLoose = (abs(genpdgid[idx])==11).any()

            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            hadronicParent = maxParent>25

            #####
            # 2. DEFINE VARIABLES
            # a genuine photon is a reconstructed photon which is matched to a generator level photon, and does not have a hadronic parent
            isGenPhoLoose = ?
            # a hadronic photon is a reconstructed photon which is matched to a generator level photon, but has a hadronic parent
            isHadPhoLoose = ?
            # a misidentified electron is a reconstructed photon which is 
            isMisIDeleLoose = ?
            # a hadronic/fake photon is a reconstructed photon that does not fall within any of the above categories
            isHadFakeLoose = ?

            #define integer definition for the photon category axis
            phoCategoryLoose = 1*isGenPhoLoose + 2*isMisIDeleLoose + 3*isHadPhoLoose + 4*isHadFakeLoose
        """

        ################
        # EVENT WEIGHTS
        ################

        #create a processor Weights object, with the same length as the number of events in the chunk
        weights = processor.Weights(len(df['event']))
  
        if not isData:

            lumiWeight = np.ones(df.size)
            nMCevents = self.mcEventYields[datasetFull]
            xsec = crossSections[dataset]
            luminosity = 35860.0
            lumiWeight *= xsec * luminosity / nMCevents

            weights.add('lumiWeight',lumiWeight)

            # PART 4: Uncomment to add weights and systematics
            """
            nPUTrue = df['Pileup_nTrueInt']

            # 4. SYSTEMATICS
            # calculate pileup weights and variations
            # use the puLookup, puLookup_Up, and puLookup_Down lookup functions to find the nominal and up/down systematic weights
            # the puLookup function is called with the full dataset name (datasetFull) and the number of true interactions
            puWeight = ?
            puWeight_Up = ?
            puWeight_Down = ?
            # add the puWeight and it's uncertainties to the weights container
            weights.add('puWeight',weight=?, weightUp=?, weightDown=?)


            eleID = self.ele_id_sf(tightElectron.eta, tightElectron.pt)
            eleIDerr = self.ele_id_err(tightElectron.eta, tightElectron.pt)
            eleRECO = self.ele_reco_sf(tightElectron.eta, tightElectron.pt)
            eleRECOerr = self.ele_reco_err(tightElectron.eta, tightElectron.pt)

            eleSF = (eleID*eleRECO).prod()
            eleSF_up = ((eleID + eleIDerr) * (eleRECO + eleRECOerr)).prod()
            eleSF_down = ((eleID - eleIDerr) * (eleRECO - eleRECOerr)).prod()
            # 4. SYSTEMATICS
            # add electron efficiency weights to the weight container
            weights.add('eleEffWeight',weight=?, weightUp=?, weightDown=?)

            muID = self.mu_id_sf(tightMuon.eta, tightMuon.pt)
            muIDerr = self.mu_id_err(tightMuon.eta, tightMuon.pt)
            muIso = self.mu_iso_sf(tightMuon.eta, tightMuon.pt)
            muIsoerr = self.mu_iso_err(tightMuon.eta, tightMuon.pt)
            muTrig = self.mu_iso_sf(abs(tightMuon.eta), tightMuon.pt)
            muTrigerr = self.mu_iso_err(abs(tightMuon.eta), tightMuon.pt)
            
            muSF = (muID*muIso*muTrig).prod()
            muSF_up = ((muID + muIDerr) * (muIso + muIsoerr) * (muTrig + muTrigerr)).prod()
            muSF_down = ((muID - muIDerr) * (muIso - muIsoerr) * (muTrig - muTrigerr)).prod()

            # 4. SYSTEMATICS
            # add electron efficiency weights to the weight container
            weights.add('muEffWeight',weight=?, weightUp=?, weightDown=?)

            #btag key name
            #name / working Point / type / systematic / jetType
            #  ... / 0-loose 1-medium 2-tight / comb,mujets,iterativefit / central,up,down / 0-b 1-c 2-udcsg 

            bJetSF_b = self.evaluator['btag2016DeepCSV_1_comb_central_0'](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btag)
            bJetSF_c = self.evaluator['btag2016DeepCSV_1_comb_central_1'](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btag)
            bJetSF_udcsg = self.evaluator['btag2016DeepCSV_1_incl_central_2'](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btag)

            bJetSF_b_up = self.evaluator['btag2016DeepCSV_1_comb_up_0'](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btag)
            bJetSF_c_up = self.evaluator['btag2016DeepCSV_1_comb_up_1'](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btag)
            bJetSF_udcsg_up = self.evaluator['btag2016DeepCSV_1_incl_up_2'](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btag)

            bJetSF_b_down = self.evaluator['btag2016DeepCSV_1_comb_down_0'](tightJets[tightJets.hadFlav==5].eta, tightJets[tightJets.hadFlav==5].pt, tightJets[tightJets.hadFlav==5].btag)
            bJetSF_c_down = self.evaluator['btag2016DeepCSV_1_comb_down_1'](tightJets[tightJets.hadFlav==4].eta, tightJets[tightJets.hadFlav==4].pt, tightJets[tightJets.hadFlav==4].btag)
            bJetSF_udcsg_down = self.evaluator['btag2016DeepCSV_1_incl_down_2'](tightJets[tightJets.hadFlav==0].eta, tightJets[tightJets.hadFlav==0].pt, tightJets[tightJets.hadFlav==0].btag)

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

            weights.add('btagWeight',btagWeight)

            weights.add('btagWeight_heavy',weight=np.ones_like(btagWeight), weightUp=btagWeight_b_up, weightDown=btagWeight_b_down)
            weights.add('btagWeight_light',weight=np.ones_like(btagWeight), weightUp=btagWeight_l_up, weightDown=btagWeight_l_down)




            #in some samples, generator systemtatics are not available, in those case the systematic weights of 1. are used
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

                #PDF Uncertainty weights
                #avoid errors from 0/0 division
                if (LHEPdfWeights[:,:1]==0).any():
                    LHEPdfWeights[:,0][LHEPdfWeights[:,0]==0] = 1.
                LHEPdfVariation = LHEPdfWeights / LHEPdfWeights[:,:1]

                weights.add('PDF', weight=np.ones(df.size), weightUp=LHEPdfVariation.max(axis=1), weightDown=LHEPdfVariation.min(axis=1))

                #Q2 Uncertainty weights
                if nLHEScaleWeights.mean()==9:
                    scaleWeightSelector=[0,1,3,5,7,8]
                elif nLHEScaleWeights.mean()==44:
                    scaleWeightSelector=[0,5,15,24,34,39]
                else:
                    scaleWeightSelector=[]

                LHEScaleVariation = LHEScaleWeights[:,scaleWeightSelector]

                weights.add('Q2Scale', weight=np.ones(df.size), weightUp=LHEScaleVariation.max(axis=1), weightDown=LHEScaleVariation.min(axis=1))

                #ISR / FSR uncertainty weights
                if not (generatorWeight==LHEWeight_originalXWGTUP).all():
                    PSWeights = PSWeights * LHEWeight_originalXWGTUP / generatorWeight

                weights.add('ISR',weight=np.ones(df.size), weightUp=PSWeights[:,2], weightDown=PSWeights[:,0])
                weights.add('FSR',weight=np.ones(df.size), weightUp=PSWeights[:,3], weightDown=PSWeights[:,1])

            else:
                weights.add('ISR',    weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))
                weights.add('FSR',    weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))
                weights.add('PDF',    weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))
                weights.add('Q2Scale',weight=np.ones(df.size),weightUp=np.ones(df.size),weightDown=np.ones(df.size))

            """

        ###################
        # FILL HISTOGRAMS
        ###################
        # PART 3: Uncomment to add histograms
        """
        #list of systematics
        systList = ['nowegiht','nominal']

        # PART 4: SYSTEMATICS
        # uncomment the full list after systematics have been implemented
        #systList = ['noweight','nominal','puWeightUp','puWeightDown','muEffWeightUp','muEffWeightDown','eleEffWeightUp','eleEffWeightDown','btagWeight_lightUp','btagWeight_lightDown','btagWeight_heavyUp','btagWeight_heavyDown', 'ISRUp', 'ISRDown', 'FSRUp', 'FSRDown', 'PDFUp', 'PDFDown', 'Q2ScaleUp', 'Q2ScaleDown']

        if not self.jetSyst=='nominal':
            systList=[self.jetSyst]

        if isData:
            systList = ['noweight']

        for syst in systList:
            
            #find the event weight to be used when filling the histograms
            weightSyst = syst
            #in the case of 'nominal', or the jet energy systematics, no weight systematic variation is used (weightSyst=None)
            if syst in ['nominal','JERUp','JERDown','JESUp','JESDown']:
                weightSyst=None
                
            if syst=='noweight':
                evtWeight = np.ones(df.size)
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
                #  use the selection.all() method to select events passing the lepton selection, 4-jet 1-tag jet selection, and either the one-photon or loose-photon selections
                #  ex: selection.all( *('LIST', 'OF', 'SELECTION', 'CUTS') )
                phosel = selection.all( *(???))
                phoselLoose = selection.all( *(???) )

                # 3. FILL HISTOGRAMS
                #    fill photon_pt and photon_eta, using the tightPhotons array, from events passing the phosel selection
                output['photon_pt'].fill(dataset=dataset,
                                         pt=?,
                                         category=?,
                                         lepFlavor=lepton,
                                         systematic=syst,
                                         weight=?)
    
                output['photon_eta'].fill(dataset=dataset,
                                         pt=?,
                                         category=?,
                                         lepFlavor=lepton,
                                         systematic=syst,
                                         weight=?)

                #    fill photon_chIso histogram, using the loosePhotons array (photons passing all cuts, except the charged hadron isolation cuts)
                output['photon_chIso'].fill(dataset=dataset,
                                            chIso=?,
                                            category=?,
                                            lepFlavor=lepton,
                                            systematic=syst,
                                            weight=?)
                
                #    fill M3 histogram, for events passing the phosel selection
                output['M3'].fill(dataset=dataset,
                                  M3=?,
                                  category=?,
                                  lepFlavor=lepton,
                                  systematic=syst,
                                  weight=?)

                
            
            # 3. GET HISTOGRAM EVENT SELECTION
            #  use the selection.all() method to select events passing the eleSel or muSel selection, 3-jet 0-btag selection, and have exactly one photon
            phosel_3j0t_e  = selection.all( *('eleSel', ???) )
            phosel_3j0t_mu = selection.all( *('muSel', ???) )

            # 3. FILL HISTOGRAMS
            # fill photon_lepton_mass_3j0t histogram, using the egammaMass array, for events passing the phosel_3j0t_e 
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=?,
                                                   category=?
                                                   lepFlavor='electron',
                                                   systematic=syst,
                                                   weight=?)
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=?,
                                                   category=?,
                                                   lepFlavor='muon',
                                                   systematic=syst,
                                                   weight=?)
            
        """

        output['EventCount'] = len(df['event'])

        return output

    def postprocess(self, accumulator):
        return accumulator



