import sys
import json
import click

import numpy as np

sys.path.append('styll_eval_metrics/')
from evaluation_metrics import joint_score
from tqdm import tqdm


def choose_best(
    original_text, target_text, transferred, embedding_type='uar', alternative_sim=False
):
    scores = []
    for i, text in enumerate(transferred):
        j_score, _ = joint_score(
            source_texts=[original_text],
            target_texts=[target_text],
            style_transferred_texts=[text],
            embedding_type=embedding_type,
            alternative_sim=alternative_sim,
        )
        scores.append(j_score)

    best_idx = np.argmax(scores)
    return transferred[best_idx]


@click.command()
@click.option('--input_path', help='path to input jsonl file', required=True)
@click.option('--do_rerank', is_flag=True)
@click.option('--rerank_alt_sim', is_flag=True)
@click.option(
    '--is_chatgpt', is_flag=True
)  # changes inputs format. Also used for ParaGuide
def main(input_path, do_rerank, rerank_alt_sim, is_chatgpt):
    input_path = input_path

    if do_rerank and not rerank_alt_sim:
        out_path = input_path.replace(".jsonl", "_reranked_scores.json")
    elif do_rerank and rerank_alt_sim:
        out_path = input_path.replace(".jsonl", "_reranked_alt_sim_scores.json")
    else:
        out_path = input_path.replace(".jsonl", "_scores.json")

    out_path = out_path + '-just-first_5'

    inferences_path = out_path.replace('.json', 'inferences.jsonl')

    author_pair_data = {}

    with open(input_path, "r") as f:
        for l in f:
            data = json.loads(l)
            source_author = data["source_author"]
            target_author = data["target_author"]

            pair = f"{source_author}->{target_author}"

            if pair not in author_pair_data:
                author_pair_data[pair] = {
                    "source_texts": [],
                    "target_texts": data['target_author_texts'],
                    "style_transferred_texts": [],
                }

            author_pair_data[pair]["source_texts"].append(data["source_text"])

            if is_chatgpt:
                result = data['output']
                while isinstance(result, (dict, list)):
                    while isinstance(result, dict):
                        # import pdb; pdb.set_trace()
                        keys = list(result.keys())
                        if len(keys) > 1:
                            if 'text' not in keys:
                                print(f'Warning: text key not in in dict: {keys}')

                                assert 'comments' in keys
                                result = result['comments']
                            else:
                                result = result['text']
                        else:
                            result = result[keys[0]]
                    while isinstance(result, list):
                        result = result[0]
                data["output"] = [result]

            if isinstance(data["output"], str):
                data["output"] = [data["output"]]

            author_pair_data[pair]["style_transferred_texts"].append(data["output"])

    print('Computing metrics')

    with open(inferences_path, 'w+') as infer_out_:
        all_scores = []
        for pair, pair_data in tqdm(list(author_pair_data.items())):
            source_texts = pair_data["source_texts"]
            target_texts = pair_data["target_texts"]
            style_transferred_texts = pair_data["style_transferred_texts"]

            if do_rerank:
                chosen_transferred = []
                for i in list(range(len(source_texts))):
                    # pair up target texts, though this may not be the best choice

                    print('WARNING: FILTERED DOWN TO FIRST 5')

                    chosen_transferred.append(
                        choose_best(
                            source_texts[i],
                            target_texts[i],
                            style_transferred_texts[i][:5],
                            embedding_type='style',
                            alternative_sim=rerank_alt_sim,
                        )
                    )

            else:
                chosen_transferred = [x[0] for x in style_transferred_texts]

            if len(target_texts) > len(source_texts):
                print(
                    f'Warning: target texts ({len(target_texts)}) are more than source texts ({len(source_texts)})...resizing'
                )
                target_texts = target_texts[: len(source_texts)]

            assert len(source_texts) == len(chosen_transferred) == len(target_texts)

            scores = joint_score(
                source_texts=source_texts,
                target_texts=target_texts,
                style_transferred_texts=chosen_transferred,
                embedding_type='uar',
            )

            all_scores.append(scores)

            inferences_data = {
                'pair': pair,
                'source_texts': source_texts,
                'target_texts': target_texts,
                'transferred_texts': style_transferred_texts,
                'selected_texts': chosen_transferred,
            }
            infer_out_.write(json.dumps(inferences_data) + '\n')

    # save all scores
    with open(out_path, "w") as f:
        json.dump(all_scores, f)

    joint_scores = np.mean([x[0] for x in all_scores])
    away_scores = np.mean([x[1]["away"] for x in all_scores])
    towards_scores = np.mean([x[1]["towards"] for x in all_scores])
    sim_scores = np.mean([x[1]["sim"] for x in all_scores])

    with open(out_path.replace(".json", "_avg.json"), "w") as f:
        json.dump(
            {
                "joint_score": joint_scores,
                "away_score": away_scores,
                "towards_score": towards_scores,
                "sim_score": sim_scores,
                "n": len(all_scores),
            },
            f,
        )


if __name__ == "__main__":
    main()
