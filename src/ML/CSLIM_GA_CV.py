"""
Use Genetic Algorithm for parametrization of the CSLIM algorithm,
I cannot try all combination of the parameters.
I need a smart way!
"""
from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.ML.CSLIM_parallel import *
# GA STUFF
from deap import base
from deap import creator
from deap import tools

# Logging stuff
import logging

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='cslim.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)


def main():
    ######################
    # Genetic algorithm
    ######################
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generator
    # alfa drawn with uniform probability between 0.1 and 3
    toolbox.register("alfa", random.uniform, 0.1, 3)
    # L1 reg with uniform prob between 1e-8 and 1e-2
    toolbox.register("l1_reg", random.uniform, 1e-8, 1e-2)
    # L2 is the same as L1 but smaller
    toolbox.register("l2_reg", random.uniform, 1e-9, 1e-3)
    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.alfa, toolbox.l1_reg, toolbox.l2_reg), n=1)
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # register the goal / fitness function
    toolbox.register("evaluate", evalOneMax)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # operator for selecting individuals for breeding the next
    # generation
    toolbox.register("select", tools.selTournament, tournsize=3)

    # create an initial population
    pop = toolbox.population(n=20)

    logging.info(" Created %i individuals" % len(pop))

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    logging.info("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    logging.info("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # try 1000 iterations
    # EVOLUTION!
    while g < 1000:
        # A new generation
        g = g + 1
        logging.info("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        logging.info("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        logging.info("  Min %s" % min(fits))
        logging.info("  Max %s" % max(fits))
        logging.info("  Avg %s" % mean)
        logging.info("  Std %s" % std)

        best_ind = tools.selBest(pop, 1)[0]
        logging.info("Best individual so far is %s, %s" % (best_ind, best_ind.fitness.values))

    logging.info("-- End of evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    logging.info("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    alfa = best_ind[0]
    l1_reg = best_ind[1]
    l2_reg = best_ind[2]
    # export csv
    cslim = SLIM(l1_reg, l2_reg, alfa)
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    cslim.fit(urm,
              tg_tracks,
              tg_playlist,
              ds)
    recs = cslim.predict()
    with open('submission_cslim.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for k in tg_playlist:
            track_ids = ''
            for r in recs[k]:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': k,
                             'track_ids': track_ids[:-1]})


# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    # extract params from individual
    alfa = individual[0]
    l1_reg = individual[1]
    l2_reg = individual[2]
    logging.info("Trying " + str(alfa) + ' ' + str(l1_reg) + ' ' + str(l2_reg))
    # create all and evaluate
    ds = Dataset()
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cslim = SLIM(l1_reg, l2_reg, alfa)
        cslim.fit(urm, tg_tracks, tg_playlist, ds)
        recs = cslim.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    return [map_at_five]


if __name__ == '__main__':
    main()
