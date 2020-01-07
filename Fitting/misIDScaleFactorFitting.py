from ROOT import TFile, TFractionFitter, TObjArray

import pprint

_file = TFile("../RootFiles/MisID_Output_electron.root")


systematics  = ["nominal",
                "FSRDown",
                "FSRUp",
                "ISRDown",
                "ISRUp",
                "JERDown",
                "JERUp",
                "JESDown",
                "JESUp",
                "PDFDown",
                "PDFUp",
                "Q2ScaleDown",
                "Q2ScaleUp",
                "btagWeight_heavyDown",
                "btagWeight_heavyUp",
                "btagWeight_lightDown",
                "btagWeight_lightUp",
                "eleEffWeightDown",
                "eleEffWeightUp",
                "muEffWeightDown",
                "muEffWeightUp",
                "puWeightDown",
                "puWeightUp",
]

results = {}

data = _file.Get("dataObs")
    
for syst in systematics:

    misID = _file.Get(f"MisIDele_{syst}")
    otherMC = _file.Get(f"Other_{syst}")
    otherMC.Add(_file.Get(f"WGamma_{syst}"))
    otherMC.Add(_file.Get(f"ZGamma_{syst}"))


    mc = TObjArray(2)
    mc.Add(misID)
    mc.Add(otherMC)

    fit = TFractionFitter(data, mc,"q")
    
    status = int(fit.Fit())

    fitResults = fit.GetFitter().Result().Parameters()

    misIDSF  = data.Integral()*fitResults[0]/mc[0].Integral()
    if not status==0:
        print (f"Error in fit while processing {syst} sample: exit status {status}")

    results[syst] = misIDSF

    del fit

pp = pprint.PrettyPrinter(indent=4)
pprint.pprint(results)
