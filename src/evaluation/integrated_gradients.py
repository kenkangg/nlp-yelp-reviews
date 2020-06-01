# from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients


def get_attributions(model, text):
    """

    Returns:
        - tokens: An array of tokens
        - attrs: An array of attributions, of same size as 'tokens',
          with attrs[i] being the attribution to tokens[i]

     """

    # tokenize text
    tokenized = tokenizer.encode_plus(text, pad_to_max_length=True, max_length=512)
    input_ids = torch.tensor(tokenized['input_ids']).to(device)
    input_ids = input_ids.view((1,-1))

    tokenized = [x for x in tokenized['input_ids'] if x!= 0]
    tokenized_text = tokenizer.convert_ids_to_tokens(tokenized)


    lig = LayerIntegratedGradients(model, model.bert.embeddings)
    attributions,delta = lig.attribute(input_ids, internal_batch_size=10, return_convergence_delta=True)

    attributions = attributions.sum(dim=-1)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions[0][:len(tokenized)]


    return tokenized_text, attributions,delta
