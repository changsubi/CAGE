"""Parameter counts for the efficiency table (Table 3)."""
import sys
sys.path.append('GroundingDINO')
from groundingdino.util.base_api import load_model
from regraph.step_a import StepA
from regraph.modules import REGraphDecoder

m = load_model('GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
               './groundingdino_swint_ogc.pth')
sa = StepA(256)
dec = REGraphDecoder(256)

backbone = sum(p.numel() for p in m.parameters())
step_a = sum(p.numel() for p in sa.parameters())
decoder = sum(p.numel() for p in dec.parameters())
head = step_a + decoder

# GroundingDINO's own detection decoder + query/box heads = exactly what CAGE removes.
# Count by parameter-name match so the efficiency table can state the real number.
def _match(model, keys):
    return sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in keys))
gd_decoder = _match(m, ['transformer.decoder'])
gd_heads = _match(m, ['bbox_embed', 'class_embed', 'refpoint_embed', 'tgt_embed', 'query'])

print("GroundingDINO backbone+enhancer : %10d  (%.1fM)" % (backbone, backbone / 1e6))
print("StepA (node construction)       : %10d  (%.3fM)" % (step_a, step_a / 1e6))
print("CAGE graph head (decoder)       : %10d  (%.3fM)" % (decoder, decoder / 1e6))
print("CAGE task head (StepA+decoder)  : %10d  (%.3fM)" % (head, head / 1e6))
print("CAGE total (backbone+head)      : %10d  (%.1fM)" % (backbone + head, (backbone + head) / 1e6))
print("-" * 60)
print("GD detection decoder (REMOVED)  : %10d  (%.2fM)" % (gd_decoder, gd_decoder / 1e6))
print("GD query/box heads (REMOVED)    : %10d  (%.2fM)" % (gd_heads, gd_heads / 1e6))
print("GD decoder+heads that CAGE drops: %10d  (%.2fM)" % (gd_decoder + gd_heads, (gd_decoder + gd_heads) / 1e6))
print("   -> CAGE head %.2fM replaces %.2fM  (%.1fx smaller)"
      % (head / 1e6, (gd_decoder + gd_heads) / 1e6, (gd_decoder + gd_heads) / max(head, 1)))
