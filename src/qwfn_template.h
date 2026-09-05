// qwfn_template -- the chat framing, in one place.
//
// Shared by qwfn-chat and qwfn-server so the two cannot drift. Everything here
// follows the template embedded in the GGUF (tokenizer.chat_template), not a
// guess at it. Two details it is easy to get wrong:
//
//   * The generation prompt pre-fills the model INTO the think block:
//     "<|im_start|>assistant\n<think>\n". Turning thinking off means emitting a
//     pre-closed "<think>\n\n</think>\n\n", not omitting the block.
//   * Reasoning effort is not a sampler knob -- it is a system message the
//     template injects. xhigh (the template's own default) and low carry text;
//     medium deliberately carries none.

#pragma once

#include <string>

namespace qwfn {

inline const char * EFFORT_XHIGH =
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.";
inline const char * EFFORT_LOW =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
    "directly to the conclusion without unnecessary elaboration.";

inline bool effort_valid(const std::string & e) {
    return e == "xhigh" || e == "medium" || e == "low" || e == "off";
}

// The leading system block, exactly as the template renders it.
// With tools, the template always opens a system turn: reasoning instructions,
// then the "# Tools" block, then the merged system text.
inline std::string build_system_block(const std::string & effort,
                                      const std::string & system_msg,
                                      const std::string & tools_block) {
    std::string reasoning;
    if (effort != "off") {
        if      (effort == "xhigh") reasoning = EFFORT_XHIGH;
        else if (effort == "low")   reasoning = EFFORT_LOW;
    }
    if (!tools_block.empty()) {
        std::string s = "<|im_start|>system\n";
        if (!reasoning.empty()) s += reasoning + "\n\n";
        s += tools_block;
        if (!system_msg.empty()) s += "\n\n" + system_msg;
        return s + "<|im_end|>\n";
    }
    if (!system_msg.empty()) {
        std::string s = "<|im_start|>system\n";
        if (!reasoning.empty()) s += reasoning + "\n\n";
        return s + system_msg + "<|im_end|>\n";
    }
    if (!reasoning.empty()) return "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    return {};
}

inline std::string build_system_block(const std::string & effort, const std::string & system_msg) {
    return build_system_block(effort, system_msg, std::string());
}

// Closes the user turn and opens the assistant turn.
inline std::string turn_tail(bool thinking) {
    std::string s = "<|im_end|>\n<|im_start|>assistant\n";
    s += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return s;
}

}  // namespace qwfn
