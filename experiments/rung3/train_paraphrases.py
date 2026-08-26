"""TRAINING paraphrase set for the task selector — DISJOINT from the gate-b eval set
(gate_b_paraphrase.py PARAPHRASES, which stays untouched as the held-out bar).
12 per task, varied syntax/vocabulary, hand-authored."""
import gate_ctx_common as gc

TRAIN_PARAPHRASES = {
    gc.PROMPT_L: [
        "go through the gate on the left, then hover over the stuffed animal",
        "fly to the left gate, pass through it, and hover over the stuffed animal",
        "navigate through the left gate and then hover above the stuffed animal",
        "cross through the gate to the left and hover over the animal toy",
        "make your way through the left gate and hover over the stuffed animal",
        "proceed through the gate located on the left and hover over the toy",
        "the gate on the left: fly through it, then hover over the stuffed animal",
        "move through the left-side gate and hover over the stuffed animal",
        "fly through the gate at the left and remain hovering over the stuffed animal",
        "traverse the left gate and hover over the stuffed toy",
        "pick the left gate, go through, hover over the stuffed animal",
        "left gate please, then hover over the stuffed animal"],
    gc.PROMPT_R: [
        "go through the gate on the right, then hover over the stuffed animal",
        "fly to the right gate, pass through it, and hover over the stuffed animal",
        "navigate through the right gate and then hover above the stuffed animal",
        "cross through the gate to the right and hover over the animal toy",
        "make your way through the right gate and hover over the stuffed animal",
        "proceed through the gate located on the right and hover over the toy",
        "the gate on the right: fly through it, then hover over the stuffed animal",
        "move through the right-side gate and hover over the stuffed animal",
        "fly through the gate at the right and remain hovering over the stuffed animal",
        "traverse the right gate and hover over the stuffed toy",
        "pick the right gate, go through, hover over the stuffed animal",
        "right gate please, then hover over the stuffed animal"],
    gc.PROMPT_CFL: [
        "go through the center gate from the left, then hover over the stuffed animal",
        "approach the center gate from the left side and fly through, hovering over the stuffed animal",
        "navigate the middle gate from the left and hover above the stuffed animal",
        "cross the central gate approaching from the left and hover over the toy",
        "from the left side, go through the center gate and hover over the stuffed animal",
        "proceed through the middle gate via the left approach and hover over the toy",
        "the center gate, from the left: fly through, then hover over the stuffed animal",
        "take the middle gate with a left approach and hover over the stuffed animal",
        "fly through the central gate coming in from the left and hover over the stuffed animal",
        "traverse the center gate from the left and hover over the stuffed toy",
        "center gate from the left, then hover over the stuffed animal",
        "using the left approach, pass the middle gate and hover over the animal"],
    gc.PROMPT_CFR: [
        "go through the center gate from the right, then hover over the stuffed animal",
        "approach the center gate from the right side and fly through, hovering over the stuffed animal",
        "navigate the middle gate from the right and hover above the stuffed animal",
        "cross the central gate approaching from the right and hover over the toy",
        "from the right side, go through the center gate and hover over the stuffed animal",
        "proceed through the middle gate via the right approach and hover over the toy",
        "the center gate, from the right: fly through, then hover over the stuffed animal",
        "take the middle gate with a right approach and hover over the stuffed animal",
        "fly through the central gate coming in from the right and hover over the stuffed animal",
        "traverse the center gate from the right and hover over the stuffed toy",
        "center gate from the right, then hover over the stuffed animal",
        "using the right approach, pass the middle gate and hover over the animal"],
}
