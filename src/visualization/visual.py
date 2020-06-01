import seaborn as sns


def visualize_confusion_matrix(cm, title=None, xlabels=[1,2,3,4,5], ylabels=[1,2,3,4,5]):
    """ Visualizes a 2-D matrix representation of a confusion matrix. Specifically for """

    assert cm.shape[0] == len(xlabels)
    assert cm.shape[1] == ylabels

    ax = sns.heatmap(cm, annot=True, xticklabels=xlabels, yticklabels=ylabels)
    ax.xaxis.set_ticks_position("top")

    plt.title(title)
    plt.show()




def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    if attr > 0:
        hue = 120
        sat = 60
        lig = 100 - int(90 * attr)
    else:
        hue = 0
        sat = 60
        lig = 100 - int(-90 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<p>"]
    tags_raw = []
    for word, importance in zip(words, importances[: len(words)]):
        new_word = format_special_tokens(word)

        color = _get_color(importance)
        if new_word!= word or new_word == "'":

          tags_raw.append((new_word, color ,1))
        else:
          tags_raw.append((new_word, color))


    for i in range(len(tags_raw)-1):
      # unwrapped = <mark style="background-color: {color}; opacity:1.0; \
      #               line-height:1.75"><font color="black"> {word}\
      #               </font></mark>'.format(
      #       color=color, word=new_word
      #   )
        skip_space = False
        if len(tags_raw[i]) == 3:

            new_word, color, _ = tags_raw[i]
            unwrapped_tag = f"<span style=\"background-color: {color}\">{new_word}</span>"
            if "'" in new_word:
                skip_space = True

            tags.append(unwrapped_tag)
        else:
            new_word, color = tags_raw[i]
            if skip_space:

                unwrapped_tag = f"<span style=\"background-color: {color}\">{new_word}</span>"
                skip_space = False
            else:
                unwrapped_tag = f"<span style=\"background-color: {color}\"> {new_word}</span>"

            tags.append(unwrapped_tag)
    tags.append("</p>")
    return ''.join(tags)


def format_special_tokens(token):
    if token.startswith("##"):
        return token.strip("##")
    if token.startswith("<") and token.endswith(">"):
        return token.strip("<>")
    return token




def visualize_text(model, text):

    tokenized, attributions, delta = get_attributions(model, text)
    tokenized = tokenized[1:-1]
    attributions = attributions[1:-1]

    formatted = format_word_importances(
        tokenized, attributions
    ),

    return formatted[0]
