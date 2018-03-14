class DAMSL:
    @staticmethod
    def sw_to_damsl(tag, prev_tag):
        if tag == "+":
            new_tag = prev_tag
        else:
            mapping_dict = {
                "sd": "statement-non-opinion",
                "b": "acknowledge",
                "sv": "statement-opinion",
                "aa": "agree-accept",
                "%-": "abandoned",
                "ba": "appreciation",
                "qy": "yes-no-question",
                "x": "non-verbal",
                "ny": "yes-answers",
                "fc": "conventional-closing",
                "%": "uninterpretable",
                "qw": "wh-question",
                "nn": "no-answers",
                "bk": "response-acknowledgement",
                "h": "hedge",
                "qy^d": "declarative-yn-question",
                "o": "other",
                "bh": "backchannel-in-question-form",
                "^q": "quotation",
                "bf": "summarize-reformulate",
                "na": "affirmative-non-yes-answers",
                "ny^e": "affirmative-non-yes-answers",
                "ad": "action-directive",
                "^2": "collaborative-completion",
                "b^m": "repeat-phrase",
                "qo": "open-question",
                "qh": "rhetorical-questions",
                "^h": "hold",
                "ar": "reject",
                "ng": "negative non-no answers",
                "nn^e": "negative non-no answers",
                "br": "signal-non-understanding",
                "no": "other-answers",
                "fp": "conventional-opening",
                "qrr": "or-clause",
                "arp": "dispreferred-answers",
                "nd": "dispreferred-answers",
                "t3": "3rd-party-talk",
                "oo": "offers-options-commits",
                "co": "offers-options-commits",
                "cc": "offers-options-commits",
                "t1": "self-talk",
                "bd": "downplayer",
                "aap": "maybe-accept-part",
                "am": "maybe-accept-part",
                "^g": "tag-question",
                "qw^d": "declarative-wh-question",
                "fa": "apology",
                "ft": "thanking"
            }
            new_tag = mapping_dict.get(tag, "%")  # mapping to literature map
            if new_tag == "%":  # mapping without rhetorical tags (see WS97 mapping guidelines for more details)
                new_tag = mapping_dict.get(
                    tag.split(",")[0].split(";")[0].split("^")[0].split("(")[0].replace("*", "").replace("@",
                                                                                                         ""),
                    "uninterpretable")
        return new_tag
