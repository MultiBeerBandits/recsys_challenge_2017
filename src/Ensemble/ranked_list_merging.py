def interleaved_merging(ranked_lists, topK, mode="skip"):
    """
    @brief      { function_description }

    @param      ranked_lists  A list of lists with the topK suggested items
                per model. The position of each list in ranked_lists defines
                the priority of its model.
    @param      mode
                "skip" - If the item to be suggested is already chosen, move
                         cursor and recompute priorities
                "continue" - If the item to be suggested is already chosen,
                             pick the first unseen item of the same list

    @return     A list with topK merged suggestions
    """
    # Initialize cursors
    curs = [0 for _ in range(len(ranked_lists))]
    merged = [None for _ in range(topK)]

    for rank in range(topK):
        selected = False

        while not selected:
            prios = _compute_priorities(curs)
            cur_i = prios.index(min(prios))

            cur = curs[cur_i]
            ranked_list = ranked_lists[cur_i]

            print("Choosing item from model {:d} at position {:d}"
                  .format(cur_i, cur))

            # Check if target item was already suggested
            try:
                item = ranked_list[cur]
                if item in merged:
                    if mode == "skip":
                        curs[cur_i] += 1
                    elif mode == "continue":
                        cur += 1
                        new_item = ranked_list[cur]
                        while new_item in merged:
                            cur += 1
                            new_item = ranked_list[cur]
                        merged[rank] = new_item
                        curs[cur_i] = cur
                        selected = True
                    else:
                        raise ValueError
                else:
                    # Suggest selected target item
                    merged[rank] = ranked_list[cur]

                    # Increment cursor of selected model
                    curs[cur_i] += 1
                    selected = True
            except IndexError:
                # This way we skip this iteration and recompute priorities
                curs[cur_i] += 1
        print("Current merged ranked list: {}".format(merged))
    return merged


def _compute_priorities(cursors):
    priorities = [0 for _ in range(len(cursors))]

    # Index is the model priority
    # Pos is the position of cursor in the ranked list at INDEX
    for index, pos in enumerate(cursors):
        priorities[index] = (index + 1) * (10 ** pos)

    return priorities


def borda_count_ensemble(ranked_lists, params, at=5):
    """
    @brief      Mix the list using the borda count voting rule
                (compute a score for each item based on its position)
                For now if an item isn't present in a list it counts as zero.

    @param      ranked_lists  A list of lists with the topK suggested items
                per model.
    @param      params
                For each model its weight

    @return     A list with topK merged suggestions
    """

    # map containing for each track id its score computed as sum_i{w_i*p_it}
    # wi is the weight of model i and pit is the position in model i of track t
    track_score = {}

    # compute for each track its score
    for i, ranked_list in enumerate(ranked_lists):

        # reverse the list since its a better way to get its position
        ranked_list = ranked_list[::-1]
        for tr_id in ranked_list:

            # update the current score of this track
            track_score[tr_id] = track_score[tr_id] + (ranked_list.index(tr_id) + 1) * params[i]

    # now we need to build a list from the track_score map according to the value
    from operator import itemgetter

    track_ordered = sorted(track_score.items(), key=itemgetter(1))

    # reverse the list
    track_ordered = track_ordered[::-1]

    # get only the first element for each tuple
    track_ordered = [x[0] for x in track_ordered]

    # return only the first at
    track_ordered = track_ordered[:at]

    return track_ordered
