
def updateJetP4(jet, pt=None, eta=None, phi=None, mass=None):

    if not pt is None:
        jet._content._contents['__fast_pt'] = pt.content
    if not eta is None:
        jet._content._contents['__fast_eta'] = eta.content
    if not phi is None:
        jet._content._contents['__fast_phi'] = phi.content
    if not mass is None:
        jet._content._contents['__fast_mass'] = mass.content
