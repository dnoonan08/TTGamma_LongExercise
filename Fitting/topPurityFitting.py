from ROOT import TFile, TFractionFitter, TObjArray

_file = TFile("../RootFiles/M3_Output.root")


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
                "puWeightUp"]

results = {}

data = _file.Get("dataObs")
    
for syst in systematics:

    mc = TObjArray(2)
    mc.Add(_file.Get(f"TopPair_{syst}"))
    mc.Add(_file.Get(f"NonTop_{syst}"))

    fit = TFractionFitter(data, mc,"q")
    
    status = int(fit.Fit())

    print (status)
    topPurity = fit.GetFitter().Result().Parameters()[0]
    results[syst] = topPurity
    
    print (syst, topPurity)

    del fit
