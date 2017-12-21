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
