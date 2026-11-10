"""Text-prompt vocabulary for SAM 3, built by looking at frames of IMG_2513/2514/2516/2518 and testing phrases on 3 frames.
Entries are (phrase, category label). Category labels follow assets/table_clean/registry.json ("<group>/<name>":
fruit/banana, fruit/apple, container/fruit_bowl are the library's own labels). Group = where the object belongs when the table is clean:
fruit -> fruit bowl, dish -> dishes area, trash (plastic rubbish) -> bin, recycling (cardboard, milk jug, tin) -> recycling box.

Seen in the frames: bananas, green apples, ginger root (fruit bowl), blue/white/brown mugs, pale green / light blue bowls, ribbed green
glasses, an empty food tin, blue crisp packets, clear zip bags, white plastic carrier bags, a 2.27 l plastic milk jug, flattened cardboard
sheets, a white bin with a liner, an Amazon cardboard box, a cream dustpan with handle, spilled dry pasta (macaroni).
Phrase tests (SAM 3, 3 frames): 'drinking glass' -> nothing, 'glass' -> the glasses (+ windows, off-table); 'milk bottle' -> nothing,
'plastic milk jug' -> 0.92-0.95; 'snack wrapper' 0.74 vs 'crisp packet' 0.90; 'apple' -> nothing on the green apples, 'green apple' 0.66;
'bowl of fruit' -> only the fruit bowl (0.71-0.95) while 'fruit bowl' also fires 0.3-0.57 on empty bowls; the bin: 'trash bin' 0.73 in
IMG_2513 but nothing in IMG_2514 (where 'plastic bag' fires on its liner; 'white bin', 'waste bin', 'bucket', 'rubbish bin' -> nothing);
pasta: 'pasta', 'spilled pasta', 'macaroni', 'dry pasta', 'pasta pieces', 'crumbs' -> nothing ('cereal' 0.3-0.4 on single grains) ->
pasta is measured by colour instead. 'ginger' 0.35 once -> not used.
"""
PROMPTS = [
    ("banana", "fruit/banana"),
    ("green apple", "fruit/apple"),
    ("mug", "dish/mug"),
    ("bowl", "dish/bowl"),
    ("glass", "dish/glass"),
    ("tin can", "recycling/tin_can"),
    ("crisp packet", "trash/wrapper"),
    ("plastic bag", "trash/plastic_bag"),
    ("plastic milk jug", "recycling/milk_jug"),
    ("flattened cardboard", "recycling/cardboard"),
    ("bowl of fruit", "container/fruit_bowl"),
    ("trash bin", "container/bin"),
    ("cardboard box", "container/recycling_box"),
    ("dustpan", "tool/dustpan"),
]
CATEGORY = {p: c for p, c in PROMPTS}
CONTAINERS = {"container/fruit_bowl", "container/bin", "container/recycling_box", "tool/dustpan"}
