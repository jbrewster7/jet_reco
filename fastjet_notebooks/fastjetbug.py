import awkward as ak 
import numpy as np
import fastjet
from coffea.nanoevents import NanoEventsFactory, EDM4HEPSchema
import dask_awkward as dak

events = NanoEventsFactory.from_root( 
    {"../../coffea_dev/root_files_may18/rv02-02.sv02-02.mILD_l5_o1_v02.E250-SetA.I402004" 
    ".Pe2e2h.eR.pL.n000.d_dstm_15090_*.slcio.edm4hep.root"
    :"events"},
    schemaclass=EDM4HEPSchema,
    permit_dask=True,
    metadata = {'b_field':5},
).events()

mupair = dak.combinations(events.PandoraPFOs[abs(events.PandoraPFOs.pdgId) == 13], 2, fields=["mu1", "mu2"])
pairmass = (mupair.mu1 + mupair.mu2).mass
muonsevent = dak.any(
    (pairmass > 80)
    & (pairmass < 100)
    & (mupair.mu1.charge == -mupair.mu2.charge),
    axis=1,
)

jetdef = fastjet.JetDefinition(fastjet.kt_algorithm,1)

# if events are masked before running through fastjet, event at index 2 of the array has indices out of range
print(fastjet._pyjet.ClusterSequence(events.PandoraPFOs[muonsevent], jetdef).exclusive_jets_constituent_index(njets=2).compute()[2])

# if events are masked after running through fastjet this is not a problem
print(fastjet._pyjet.ClusterSequence(events.PandoraPFOs, jetdef).exclusive_jets_constituent_index(njets=2)[muonsevent].compute()[2])
