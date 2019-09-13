# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        distance=float("-Inf")
        food = currentGameState.getFood()
        foodList = food.asList()
        "*** YOUR CODE HERE ***"
        if ( action == 'Stop') :
           return float("-Inf")

        for state in newGhostStates :
           if( state.getPosition() == newPos ) and state.scaredTimer == 0:
              return float("-Inf")
        for x in foodList :
            templacedistance=-1*manhattanDistance(newPos,x)
            if(templacedistance>distance) :
               distance=templacedistance                                                              
        return distance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValua(gameState, depth, numberGhost):
           if gameState.isWin() or depth==0 or gameState.isLose() :
              return self.evaluationFunction(gameState)
           v=-(float("inf"))
           legalState=gameState.getLegalActions(0)

           for action in legalState :
             v=max(v,minValua(gameState.generateSuccessor(0,action),depth-1,1, numberGhost))
           return v





        def minValua(gameState,depth,indexGhost,numberGhost):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)   
            v=float("inf")
            legalState=gameState.getLegalActions(indexGhost)  
            if numberGhost==indexGhost : 
                for action in legalState:
                  v=min(v,maxValua(gameState.generateSuccessor(indexGhost,action),depth-1,numberGhost))
            else :
                for action in legalState:
                  v=min(v,minValua(gameState.generateSuccessor(indexGhost,action),depth-1,indexGhost+1,numberGhost))
            return v   
        depth=self.depth*gameState.getNumAgents()
        legalState=gameState.getLegalActions(0)
        numberGhost = gameState.getNumAgents() - 1
        bestaction = Directions.STOP
        score=-(float("inf"))
        for action in legalState:
            nextState = gameState.generateSuccessor(0, action)
            previousScore=score
            score=max(score ,minValua(nextState,depth-1,1, numberGhost) )
            if score>previousScore:
               bestaction=action
        return  bestaction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxAB(gameState,alpha,beta,depth,numberGhost) :
           if gameState.isWin() or gameState.isLose() or depth==0 :
              return self.evaluationFunction(gameState)
           v=-(float("inf"))
           legalAction=gameState.getLegalActions(0)
           for action in legalAction:
              v=max(v,minAB(gameState.generateSuccessor(0,action),alpha,beta,depth-1,numberGhost,1) )
              if v > beta :
                  return v
              alpha=max(alpha,v)
           return v





        def minAB(gameState,alpha,beta,depth,numberGhost,indexGhost) :
           if gameState.isWin() or gameState.isLose() or depth==0 :
              return self.evaluationFunction(gameState)
           v=float("inf")
           legalAction=gameState.getLegalActions(indexGhost)
           if indexGhost==numberGhost:
               for action in legalAction:
                  v=min(v,maxAB(gameState.generateSuccessor(indexGhost,action),alpha,beta,depth-1,numberGhost))
                  if  v < alpha :
                        return v
                  beta=min(beta,v)
           else :
               for action in legalAction:
                  v=min(v,minAB(gameState.generateSuccessor(indexGhost,action),alpha,beta,depth-1,numberGhost,indexGhost+1))
                  if  v < alpha :
                        return v
                  beta=min(beta,v)
           return v
        depth=self.depth*gameState.getNumAgents()
        legalState=gameState.getLegalActions(0)
        numberGhost = gameState.getNumAgents() - 1
        alpha=-float("inf")
        beta=float("inf")
        bestaction = Directions.STOP
        score=-float("inf")
        for action in legalState:
            nextState = gameState.generateSuccessor(0, action)
            previousScore=score
            score=max(score, minAB(nextState,alpha,beta,depth-1,numberGhost,1) )
            if score>previousScore:
               bestaction=action
            if score > beta:
                return bestaction
            alpha = max(alpha, score)
               
        return  bestaction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxE(gameState, depth, numberGhost):
           if gameState.isWin() or depth==0 or gameState.isLose() :
              return self.evaluationFunction(gameState)
           v=-(float("inf"))
           legalState=gameState.getLegalActions(0)
           for action in legalState :
             v=max(v,minE(gameState.generateSuccessor(0,action),depth-1,1, numberGhost))
           return v





        def minE(gameState,depth,indexGhost,numberGhost):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)   
            v=0.0
            legalState=gameState.getLegalActions(indexGhost)
            azly=len( legalState )
            p=1.0/azly 
            if numberGhost==indexGhost : 
                for action in legalState:
                  v=v+p*(maxE(gameState.generateSuccessor(indexGhost,action),depth-1,numberGhost))
            else :
                for action in legalState:
                  v=v+p*(minE(gameState.generateSuccessor(indexGhost,action),depth-1,indexGhost+1,numberGhost)) 
            return v   
        depth=self.depth*gameState.getNumAgents()
        legalState=gameState.getLegalActions(0)
        numberGhost = gameState.getNumAgents() - 1
        bestaction = Directions.STOP
        score=-(float("inf"))
        for action in legalState:
            nextState = gameState.generateSuccessor(0, action)
            previousScore=score
            score = max(score ,minE(nextState,depth-1,1, numberGhost) )
            if score>previousScore:
               bestaction=action
        return  bestaction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
