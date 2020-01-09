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
        
        ###
        self._accumulator = processor.dict_accumulator({
            ##photon histograms
            'photon_pt': hist.Hist("Counts", dataset_axis, pt_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_eta': hist.Hist("Counts", dataset_axis, eta_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_chIso': hist.Hist("Counts", dataset_axis, chIso_axis, phoCategory_axis, lep_axis, systematic_axis),

            'photon_lepton_mass': hist.Hist("Counts", dataset_axis, mass_axis, phoCategory_axis, lep_axis, systematic_axis),
            'photon_lepton_mass_3j0t': hist.Hist("Counts", dataset_axis, mass_axis, phoCategory_axis, lep_axis, systematic_axis),

            'M3'      : hist.Hist("Counts", dataset_axis, m3_axis, phoCategory_axis, lep_axis, systematic_axis),
            'M3Presel': hist.Hist("Counts", dataset_axis, m3_axis, lep_axis, systematic_axis),

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

        year=2016

        ## apply triggers
        # muon events should be triggered by either the HLT_IsoMu24 or HLT_IsoTkMu24 triggers
        # electron events should be triggered by HLT_Ele27_WPTight_Gsf trigger
        # HINT: trigger values can be accessed with the variable df['TRIGGERNAME'], 
        # the bitwise or operator can be used to select multiple triggers df['TRIGGER1'] | df['TRIGGER2']
        muTrigger = df['HLT_IsoMu24'] | df['HLT_IsoTkMu24']
        eleTrigger = df['HLT_Ele27_WPTight_Gsf']


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
            genJet = JaggedCandidateArray.candidatesfromcounts(
                df['nGenJet'],
                pt = df['GenJet_pt'],
                eta = df['GenJet_eta'],
                phi = df['GenJet_phi'],
                mass = df['GenJet_mass'],
            )

            # fix what seems to be a bug, genJets get skimmed after the genJet matching:
            #   jets matched to a genJet with pt<10 still get assigned a value for Jet_genJetIdx, but that index is not present in the
            #   genJet list because it is cut.  In these cases, set idx to -1
            jets.genJetIdx[jets.genJetIdx>=genJet.counts] = -1

            jets['ptGenJet'][jets.genJetIdx>-1] = genJet[jets.genJetIdx[jets.genJetIdx>-1]].pt
            jets['rho'] = jets.pt.ones_like()*rho

            #adds additional columns to the jets array, containing the jet pt with JEC and JER variations
            Jet_transformer.transform(jets, forceStochastic=False)
            

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
            


    
        #select tight muons
        muonSelectTight = ((muons.pt>30) & 
                           (abs(muons.eta)<2.4) & 
                           (muons.tightId) & 
                           (muons.relIso < 0.15)
                          )

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
        electronSelectTight = ((electrons.pt>35) & 
                               (abs(electrons.eta)<2.1) & 
                               eleEtaGap &      
                               (electrons.cutBased>=4) &
                               elePassD0 & 
                               elePassDZ
                              )

        #select loose electrons
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


        #count events which have exactly one muon
        oneMuon = (tightMuon.counts == 1)
        muVeto = (tightMuon.counts == 0)

        #count events which have exactly one electron
        oneEle = (tightElectron.counts == 1)
        eleVeto = (tightElectron.counts == 0)

        #count events with no loose leptons
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


        #parse VID cuts, define loose photons (photons without chIso cut)
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
        
        ##medium jet ID cut
        jetIDbit = 1

        ##check dR jet,lepton & jet,photon
        jetMu = jets['p4'].cross(tightMuon['p4'],nested=True)
        dRjetmu = ((jetMu.i0.delta_r(jetMu.i1)).min()>0.4) | (tightMuon.counts==0)

        jetEle = jets['p4'].cross(tightElectron['p4'],nested=True)
        dRjetele = ((jetEle.i0.delta_r(jetEle.i1)).min()>0.4) | (tightElectron.counts==0)

        jetPho = jets['p4'].cross(tightPhotons['p4'],nested=True)
        dRjetpho = ((jetPho.i0.delta_r(jetPho.i1)).min()>0.1) | (tightPhotons.counts==0)


        if not isData:
            if self.jetSyst=='JERUp':
                updateJetP4(jets, pt = jets.pt_jer_up, mass = jets.mass_jer_up)
            if self.jetSyst=='JERDown':
                updateJetP4(jets, pt = jets.pt_jer_down, mass = jets.mass_jer_down)
            if self.jetSyst=='JESUp':
                updateJetP4(jets, pt = jets.pt_jes_up, mass = jets.mass_jes_up)
            if self.jetSyst=='JESDown':
                updateJetP4(jets, pt = jets.pt_jes_down, mass = jets.mass_jes_down)

        jetSelect = ((jets.pt > 30) &
                     (abs(jets.eta) < 2.4) &
                     ((jets.jetId >> jetIDbit & 1)==1) &
                     dRjetmu & dRjetele & dRjetpho )
        
        tightJets = jets[jetSelect]

        #find jets passing DeepCSV medium working point
        bTagWP = 0.6321   #2016 DeepCSV working point
        btagged = tightJets.btag>bTagWP

        bTaggedJets = tightJets[btagged]

        ## Define M3, mass of 3-jet pair with highest pT
        triJet = tightJets['p4'].choose(3)
        triJetPt = (triJet.i0 + triJet.i1 + triJet.i2).pt
        triJetMass = (triJet.i0 + triJet.i1 + triJet.i2).mass
        M3 = triJetMass[triJetPt.argmax()]

        leadingMuon = tightMuon[:,:1] 
        leadingElectron = tightElectron[:,:1]        
        
        leadingPhoton = tightPhotons[:,:1]
        leadingPhotonLoose = loosePhotons[:,:1]


        egamma = leadingElectron['p4'].cross(leadingPhoton['p4'])
        mugamma = leadingMuon['p4'].cross(leadingPhoton['p4'])
        egammaMass = (egamma.i0 + egamma.i1).mass
        mugammaMass = (mugamma.i0 + mugamma.i1).mass
        
        
        
        if not isData:
            #### Photon categories, using genIdx branch
            idx = leadingPhoton.genIdx
            
            # look through gen particle history, finding the highest PDG ID
            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            # reco photons matched to a generated photon
            matchedPho = (genpdgid[idx]==22).any()
            # reco photons really generated as electrons
            isMisIDele = (abs(genpdgid[idx])==11).any()

            # if the gen photon has a PDG ID > 25 in it's history, it has a hadronic parent
            hadronicParent = maxParent>25

            isGenPho = matchedPho & ~hadronicParent
            isHadPho = matchedPho & hadronicParent
            isHadFake = ~(isMisIDele | isGenPho | isHadPho) & (leadingPhoton.counts==1)
            
            #define integer definition for the photon category axis
            phoCategory = 1*isGenPho + 2*isMisIDele + 3*isHadPho + 4*isHadFake
            

            # do photon matching for loose photons as well
            # look through parentage to find if any hadrons in genPhoton parent history
            idx = leadingPhotonLoose.genIdx

            # reco photons matched to a generated photon
            matchedPhoLoose = (genpdgid[idx]==22).any()
            # reco photons really generated as electrons
            isMisIDeleLoose = (abs(genpdgid[idx])==11).any()

            maxParent = maxHistoryPDGID(idx.content, idx.starts, idx.stops, 
                                        genpdgid.content, genpdgid.starts, genpdgid.stops, 
                                        genmotherIdx.content, genmotherIdx.starts, genmotherIdx.stops)

            hadronicParent = maxParent>25

            isGenPhoLoose = matchedPhoLoose & ~hadronicParent
            isHadPhoLoose = matchedPhoLoose & hadronicParent
            isHadFakeLoose = ~(isMisIDeleLoose | isGenPhoLoose | isHadPhoLoose) & (leadingPhotonLoose.counts==1)        

            #define integer definition for the photon category axis
            phoCategoryLoose = 1*isGenPhoLoose + 2*isMisIDeleLoose + 3*isHadPhoLoose + 4*isHadFakeLoose

        else:
            phoCategory = np.ones(df.size)
            phoCategoryLoose = np.ones(df.size)
        

        # muon selection, requires events to pass:   muon trigger
        #                                            overlap removal
        #                                            have exactly one muons
        #                                            have no electrons
        #                                            have no loose muons
        #                                            have no loose electrons

        muon_eventSelection = (muTrigger & passOverlapRemoval &
                               oneMuon & eleVeto &
                               looseMuonSel & looseElectronSel)

        electron_eventSelection = (eleTrigger & passOverlapRemoval &
                                   oneEle & muVeto &
                                   looseMuonSel & looseElectronSel)

        #create a selection object
        selection = processor.PackedSelection()

        #add selection 'eleSel', for events passing the electron event selection, and muSel for those passing the muon event selection
        selection.add('eleSel',electron_eventSelection)
        selection.add('muSel',muon_eventSelection)

        #add two jet selection criteria
        #   First, 'jetSel' which selects events with at least 4 jets and at least
        selection.add('jetSel', (tightJets.counts >= 4) & (bTaggedJets.counts >= 1) )
        selection.add('jetSel_3j0t', (tightJets.counts >= 3) & (bTaggedJets.counts == 0) )

        selection.add('zeroPho', tightPhotons.counts == 0)
        selection.add('onePho', tightPhotons.counts == 1)
        selection.add('loosePho', loosePhotons.counts == 1)


        #create a processor Weights object, with the same length as the number of events in the chunk
        weights = processor.Weights(len(df['event']))
  
        if not 'Data' in dataset:

            lumiWeight = np.ones(df.size)
            nMCevents = self.mcEventYields[datasetFull]
            xsec = crossSections[dataset]
            lumiWeight *= xsec * lumis[year] / nMCevents

            weights.add('lumiWeight',lumiWeight)


            nPUTrue = df['Pileup_nTrueInt']
            puWeight = puLookup(datasetFull, nPUTrue)
            puWeight_Up = puLookup_Up(datasetFull, nPUTrue)
            puWeight_Down = puLookup_Down(datasetFull, nPUTrue)
            
            weights.add('puWeight',weight=puWeight, weightUp=puWeight_Up, weightDown=puWeight_Down)

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


        systList = ['noweight','nominal','puWeightUp','puWeightDown','muEffWeightUp','muEffWeightDown','eleEffWeightUp','eleEffWeightDown','btagWeight_lightUp','btagWeight_lightDown','btagWeight_heavyUp','btagWeight_heavyDown', 'ISRUp', 'ISRDown', 'FSRUp', 'FSRDown', 'PDFUp', 'PDFDown', 'Q2ScaleUp', 'Q2ScaleDown']

        if not self.jetSyst=='nominal':
            systList=[self.jetSyst]

        if isData:
            systList = ['noweight']

        for syst in systList:
            
            weightSyst = syst
            if syst in ['nominal','JERUp','JERDown','JESUp','JESDown']:
                weightSyst=None
                
            if syst=='noweight':
                evtWeight = np.ones(df.size)
            else:
                evtWeight = weights.weight(weightSyst)

            jetSelType = 'jetSel'

            for lepton in ['electron','muon']:
                if lepton=='electron':
                    lepSel='eleSel'
                if lepton=='muon':
                    lepSel='muSel'

                phosel = selection.all( *(lepSel, 'jetSel', 'onePho'))
                phoselLoose = selection.all( *(lepSel, 'jetSel', 'loosePho') )
                zeroPho = selection.all( *(lepSel, 'jetSel', 'zeroPho') )

                output['photon_pt'].fill(dataset=dataset,
                                         pt=tightPhotons.p4.pt[:,:1][phosel].flatten(),
                                         category=phoCategory[phosel].flatten(),
                                         lepFlavor=lepton,
                                         systematic=syst,
                                         weight=evtWeight[phosel].flatten())
    
                output['photon_eta'].fill(dataset=dataset,
                                          eta=tightPhotons.eta[:,:1][phosel].flatten(),
                                          category=phoCategory[phosel].flatten(),
                                          lepFlavor=lepton,
                                          systematic=syst,
                                          weight=evtWeight[phosel].flatten())

                
                output['photon_chIso'].fill(dataset=dataset,
                                            chIso=loosePhotons.chIso[:,:1][phoselLoose].flatten(),
                                            category=phoCategoryLoose[phoselLoose].flatten(),
                                            lepFlavor=lepton,
                                            systematic=syst,
                                            weight=evtWeight[phoselLoose].flatten())
                
                
                output['M3'].fill(dataset=dataset,
                                  M3=M3[phosel].flatten(),
                                  category=phoCategoryLoose[phosel].flatten(),
                                  lepFlavor=lepton,
                                  systematic=syst,
                                  weight=evtWeight[phosel].flatten())
                
                output['M3Presel'].fill(dataset=dataset,
                                        M3=M3[zeroPho].flatten(),
                                        lepFlavor=lepton,
                                        systematic=syst,
                                        weight=evtWeight[zeroPho].flatten())                            
    
            
            phosel_e  = selection.all( *('eleSel', 'jetSel', 'onePho') )
            phosel_mu = selection.all( *('muSel', 'jetSel', 'onePho') )

            phosel_3j0t_e  = selection.all( *('eleSel', 'jetSel_3j0t', 'onePho') )
            phosel_3j0t_mu = selection.all( *('muSel', 'jetSel_3j0t', 'onePho') )

            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=egammaMass[phosel_e].flatten(),
                                              category=phoCategory[phosel_e].flatten(),
                                              lepFlavor='electron',
                                              systematic=syst,
                                              weight=evtWeight[phosel_e].flatten())
            output['photon_lepton_mass'].fill(dataset=dataset,
                                              mass=mugammaMass[phosel_mu].flatten(),
                                              category=phoCategory[phosel_mu].flatten(),
                                              lepFlavor='muon',
                                              systematic=syst,
                                              weight=evtWeight[phosel_mu].flatten())
    
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=egammaMass[phosel_3j0t_e].flatten(),
                                                   category=phoCategory[phosel_3j0t_e].flatten(),
                                                   lepFlavor='electron',
                                                   systematic=syst,
                                                   weight=evtWeight[phosel_3j0t_e].flatten())
            output['photon_lepton_mass_3j0t'].fill(dataset=dataset,
                                                   mass=mugammaMass[phosel_3j0t_mu].flatten(),
                                                   category=phoCategory[phosel_3j0t_mu].flatten(),
                                                   lepFlavor='muon',
                                                   systematic=syst,
                                                   weight=evtWeight[phosel_3j0t_mu].flatten())
            

        output['EventCount'] = len(df['event'])

        return output

    def postprocess(self, accumulator):
        return accumulator



