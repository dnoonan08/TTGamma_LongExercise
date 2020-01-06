from ROOT import TFile, TFractionFitter, TObjArray

_file = TFile("../RootFiles/Isolation_Output.root")


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

    mc = TObjArray(2)
    mc.Add(_file.Get(f"Isolated_{syst}"))
    mc.Add(_file.Get(f"NonPrompt_{syst}"))

    fit = TFractionFitter(data, mc,"q")
    
    status = fit.Fit()

    fitResults = fit.GetFitter().Result().Parameters()

    isolatedSF  = data.Integral()*fitResults[0]/mc[0].Integral()
    nonPromptSF = data.Integral()*fitResults[1]/mc[1].Integral()

    # print(mc[0].GetBinContent(1)*isolatedSF)
    # print(mc[1].GetBinContent(1)*nonPromptSF)

    isolatedRate = mc[0].GetBinContent(1)*isolatedSF
    nonPromptRate = mc[1].GetBinContent(1)*nonPromptSF

    phoPurity = isolatedRate / (isolatedRate + nonPromptRate)

    results[syst] = phoPurity    
    print (syst, phoPurity)

    del fit
