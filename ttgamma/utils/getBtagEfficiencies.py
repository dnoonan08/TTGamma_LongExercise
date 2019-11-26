from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor

from coffea.lookup_tools import extractor, dense_lookup

import numpy as np
import pickle

btagEff_ptBins = np.array([30,50,70,100,140,200,500])

btagEff_etaBins = np.arange(0,2.4001,.6)

# Look at ProcessorABC to see the expected methods and what they are supposed to do
class BjetEfficiencies(processor.ProcessorABC):
#     def __init__(self, runNum = -1, eventNum = -1):
    def __init__(self):
        dataset_axis = hist.Cat("dataset", "Dataset")
        dataset_axis = hist.Cat("dataset", "Dataset")
        jetPt_axis = hist.Bin('jetPt','jetPt',btagEff_ptBins)
        jetEta_axis = hist.Bin('jetEta','jetEta',btagEff_etaBins)
        jetFlav_axis = hist.Bin('jetFlav','jetFlav',[0,4,5,6])
        self._accumulator = processor.dict_accumulator({

            'hJets'  : hist.Hist("Counts", dataset_axis, jetPt_axis, jetEta_axis, jetFlav_axis),
            'hBJets' : hist.Hist("Counts", dataset_axis, jetPt_axis, jetEta_axis, jetFlav_axis),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()

        datasetFull = df['dataset']
        print(datasetFull)
        if '2016' in datasetFull:
            year=2016
            yearStr='2016'
            dataset=datasetFull.replace('_2016','')
        if '2017' in datasetFull:
            year=2017
            yearStr='2017'
            dataset=datasetFull.replace('_2017','')
        if '2018' in datasetFull:
            year=2018
            yearStr='2018'
            dataset=datasetFull.replace('_2018','')

        jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'],
            eta=df['Jet_eta'],
            phi=df['Jet_phi'],
            mass=df['Jet_mass'],
            jetId=df['Jet_jetId'],
            btag=df['Jet_btagDeepB'],
            hadFlav=df['Jet_hadronFlavour'],
            genIdx=df['Jet_genJetIdx'],
        )
        
        jetSelect = ((jets.pt > 30) &
                     (abs(jets.eta) < 2.4) &
                     ((jets.jetId >> 0 & 1)==1))
        
        bTagWP = 0.6321
        if year == '2017':
            bTagWP = 0.4941
        if year == '2018':
            bTagWP = 0.4184

        Jets = jets[jetSelect]
        bJets = jets[jetSelect & (jets.btag>bTagWP)]
        output['hJets'].fill(dataset=datasetFull,
                             jetPt=Jets.pt.flatten(),
                             jetEta=abs(Jets.eta).flatten(),
                             jetFlav=Jets.hadFlav.flatten(),
                            )
        
        output['hBJets'].fill(dataset=datasetFull,
                              jetPt=bJets.pt.flatten(),
                              jetEta=abs(bJets.eta).flatten(),
                              jetFlav=bJets.hadFlav.flatten(),
                             )

        return output

    def postprocess(self, accumulator):
        return accumulator



from fileList import *
fileSet_noData = {x:fileSet_2016[x] for x in fileSet_2016 if not 'Data' in x}

f = {'QCD_Pt20to30_Ele_2016': fileSet_2016['QCD_Pt20to30_Ele_2016'],
     #'QCD_Pt300toInf_Ele_2016': fileSet_2016['QCD_Pt300toInf_Ele_2016'],
 }

output = processor.run_uproot_job(fileSet_noData,
                                  treename='Events',
                                  processor_instance=BjetEfficiencies(),
                                  executor=processor.futures_executor,
                                  executor_args={'workers': 4, 'flatten': True},
#                                   chunksize=50000,                                                                                          
#                                   maxchunks=0                                                                                              
                                 )



l_total = output['hJets'].integrate('jetFlav',slice(0,4)).values()
c_total = output['hJets'].integrate('jetFlav',slice(4,5)).values()
b_total = output['hJets'].integrate('jetFlav',slice(5,6)).values()

l_tagged = output['hBJets'].integrate('jetFlav',slice(0,4)).values()
c_tagged = output['hBJets'].integrate('jetFlav',slice(4,5)).values()
b_tagged = output['hBJets'].integrate('jetFlav',slice(5,6)).values()

for k in l_total.keys():
    l_total[k] = np.maximum(1,l_total[k])
    c_total[k] = np.maximum(1,c_total[k])
    b_total[k] = np.maximum(1,b_total[k])

    if 0 in l_total[k]: print (k, 'light', l_total[k])
    if 0 in l_total[k]: print (k, 'charm', c_total[k])
    if 0 in l_total[k]: print (k, 'b', b_total[k])

btagEff = []
samples = []
for k in l_total.keys():
    btagEff.append(np.array([l_tagged[k]/l_total[k],c_tagged[k]/c_total[k],b_tagged[k]/b_total[k]]))
    samples.append(k[0])


taggingEffLookup = dense_lookup.dense_lookup(np.array(btagEff),(samples,[0,4,5],btagEff_ptBins,btagEff_etaBins))


with open('taggingEfficienciesDenseLookup.pkl','wb') as _file:
    pickle.dump(taggingEffLookup,_file)
