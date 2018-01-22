def interleaved_merging(ranked_lists, n_picks, topK, mode="skip"):
    """
    @brief      { function_description }

    @param      ranked_lists  A list of lists with the topK suggested items
                per model. The position of each list in ranked_lists defines
                the priority of its model.
    @param      n_picks A list whose entries are the number of items to pick
                from model suggestions when it's selected.
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
    n_merged = 0

    while n_merged < topK:
        prios = _compute_priorities(curs)
        cur_i = prios.index(min(prios))

        cur = curs[cur_i]
        ranked_list = ranked_lists[cur_i]

        # Compute number of items to pick from ranked_list. It is the min among
        # number of items left in ranked_list, n_picks defined for ranked_list
        # and remaining slots in merged suggestions list
        n_pick = min(len(ranked_list) - cur, n_picks[cur_i], topK - n_merged)
        # print("Merging {:d} suggestions from model {:d}..."
        #       .format(n_pick, cur_i))
        for pick in range(n_pick):
            try:
                cur = curs[cur_i]
                item = ranked_list[cur]

                # Check if item was already suggested
                if item in merged:
                    if mode == "continue":
                        # Look for first unseen item
                        cur += 1
                        new_item = ranked_list[cur]
                        while new_item in merged:
                            cur += 1
                            new_item = ranked_list[cur]
                        # Suggest it and move cursor after
                        merged[n_merged] = new_item
                        n_merged += 1
                        curs[cur_i] = cur + 1
                    else:
                        raise ValueError
                else:
                    # Suggest selected target item
                    merged[n_merged] = ranked_list[cur]
                    n_merged += 1

                    # Increment cursor of selected model
                    curs[cur_i] += 1
            except IndexError:
                # This way we skip this iteration and recompute priorities
                curs[cur_i] += 1
                break
    # print("Current merged ranked list: {}".format(merged))
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
