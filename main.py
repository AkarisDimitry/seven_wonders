import random, time, concurrent.futures
import numpy as np
from numba import jit
from enum import Enum, auto

random.seed(10)

# Definición de tipos de recursos
class ResourceType(Enum):
    WOOD = auto()
    STONE = auto()
    CLAY = auto()
    ORE = auto()
    GLASS = auto()
    LOOM = auto()
    PAPYRUS = auto()
    MONEY = auto()

# Definición de tipos de cartas
class CardType(Enum):
    RAW_MATERIAL = auto()
    MANUFACTURED_GOOD = auto()
    CIVILIAN_STRUCTURE = auto()
    SCIENTIFIC_STRUCTURE = auto()
    COMMERCIAL_STRUCTURE = auto()
    MILITARY_STRUCTURE = auto()
    GUILD = auto()

# -------------------------------------------------------------------------- #
class Card:
    def __init__(self, name:str, card_type:object, age:int, cost:list, resources_produced:list=None, 
                        chain:list=[], points:int=0, effect:dict={},
                        militar:int=0, science:dict={},
                        players:int=0, ):
        self.name = name
        self.card_type = card_type
        self.age = age

        self.cost = cost
        self.chain = chain

        self.resources_produced = resources_produced or {}
        self.points = points
        self.effect = effect

        self.militar = militar
        self.science = science

        self.min_players = players 

    def print(self, v=1):
        cost_string = ','.join([f'{items.name:8s}:{keys}' for items, keys in self.cost.items() ])
        string = f'{self.name:20s} - ' + f'{self.card_type:30s} - '[9:] + cost_string
        print(string)
        return string 

# -------------------------------------------------------------------------- #
class Player:
    def __init__(self):
        self.cards = []
        self.cards_names = {}
        self.cards_discards = []

        self.resources = {resource: 0 for resource in ResourceType}; self.resources[ResourceType.MONEY] = 3
        self.resources_spend = {resource: 0 for resource in ResourceType}
        self.resources_tradiable =  {resource: 0 for resource in ResourceType}
        self.resources_selectable = [] # [ResourceType.WOOD, ResourceType.ORE, ResourceType.CLAY]

        self.resources_combination = self.get_resources_combination(just_selectable=False)
        self.resources_combination_just_selectable = self.get_resources_combination(just_selectable=True)

        self.points = 0
        self.wonder = None
        self.hand = None

        self.neighbors = None
        self.neighbors_cost = None

        # === Militar ===
        self.militar = 0
        self.militar_neighbors = None
        self.militar_score = {'win':0, 'loss':0, 'tie':0}
        self.militar_points = {'win':[0,1,3,5], 'loss':[0,-1,-1,-1], 'tie':[0,0,0,0]}
        self.militar_battle = {'win':0, 'loss':0, 'tie':0}

        self.science = {'Tabla':0, 'Engranaje':0, 'Compass':0, 'TEC':0}

        self.card_effects = {'FREE_DISCARD':0}
        self.score_effects = {}

        self.COMMERCIAL_STRUCTURE_effects = {}


        self.agent = 'random'

    def get_resources_combination(self, just_selectable=False):
        '''
        resource_combinations = []
        for resource_combination in itertools.product(*self.resources_selectable):
            resource_combinations.append( {key: self.resources.get(key, 0) + resource_combination.count(key)  for key in self.resources.keys()  } )
        return resource_combinations
        '''
        return [self.resources] if len(self.resources_selectable) == 0 else [ {key: self.resources.get(key, 0)*(not just_selectable) + resource_combination.count(key)  for key in self.resources.keys()  } for resource_combination in itertools.product(*self.resources_selectable) ]

    def card_is_playable(self, card):
        return not self.repeted(card) and (self.has_chain(card) or self.has_required_resources(card)) 

    def playable_cards(self, ):
        return [ self.card_is_playable(card) for card in self.hand ] 

    def can_build_wonder(self, ):
        return (not self.wonder.level_max()) and self.has_required_resources( self.wonder.buildings[self.wonder.level] )

    def repeted(self, card):
        return card.name in self.cards_names

    def has_chain(self, card):
        return sum([ch in self.cards_names for ch in card.chain ]) > 0

    def how_much_to_pay(self, resource, amount):
        sorted_indices = [index for index, _ in sorted(enumerate(self.neighbors_cost), key=lambda x: x[1])]
        requiered_money = 0

        for n_id in sorted_indices:
            if self.neighbors[n_id].resources.get(resource, 0) >= amount:
                requiered_money += amount * self.neighbors_cost[n_id]
                amount = 0

            else:
                requiered_money += self.neighbors[n_id].resources.get(resource, 0) * self.neighbors_cost[n_id]
                amount -= self.neighbors[n_id].resources.get(resource, 0)

        return  requiered_money

    def how_much_to_pay(self, resources, gready_search=True, get_neighbors_sorted=False):
        """
        Calcula el costo de obtener los recursos especificados de los vecinos.

        Args:
            resources (dict): Un diccionario que especifica los recursos requeridos y sus cantidades.
            gready_search (bool, optional): Indica si se debe realizar una búsqueda voraz (por defecto es True).

        Returns:
            list: Una lista de combinaciones de precios requeridos para obtener los recursos especificados.
        """
        # Una lista para almacenar las combinaciones de precios requeridos.
        requiered_money_combinations = []
        # Ordena los vecinos por costo.
        neighbors_sorted = [k for v, k in sorted(zip(self.neighbors_cost, self.neighbors), key=lambda x: x[0])]
        neighbors_order  = [k for v, k in sorted(zip(self.neighbors_cost, np.arange(len(self.neighbors), dtype=np.uint32) ), key=lambda x: x[0])]

        # Obtiene la combinación de recursos de cada vecino.
        list_neighbors_resources_combination = [n.resources_combination for n in neighbors_sorted]

        # Itera sobre todas las combinaciones de recursos de los vecinos.
        for nrc in itertools.product(*list_neighbors_resources_combination):  # nrc: neighbors_resources_combination, just one element
            # Un arreglo para almacenar los precios requeridos por vecino.
            requiered_money = np.zeros(len(self.neighbors))
            # Un diccionario para mantener un registro de los recursos necesarios en la iteración anterior.
            needed_resources_prev = resources
            # Itera sobre cada combinación de recursos de un vecino.
            for i, one_nrc in enumerate(nrc):
                # Calcula los nuevos recursos necesarios.
                needed_resources_new = {
                    resource: max(amount - one_nrc.get(resource, 0) - neighbors_sorted[i].resources.get(resource, 0), 0)
                    for resource, amount in needed_resources_prev.items()
                }

                # Calcula el precio requerido para este vecino.
                requiered_money[i] = (sum(needed_resources_prev.values()) - sum(needed_resources_new.values())) * sorted(self.neighbors_cost)[i]
                # Actualiza el registro de los recursos necesarios.
                needed_resources_prev = needed_resources_new

                # Si se han alcanzado todos los recursos necesarios, agrega la combinación de precios requeridos y detén el bucle.
                if sum(needed_resources_new.values()) == 0:
                    requiered_money_combinations.append(requiered_money)
                    break

            # Si la búsqueda voraz está activada, devuelve el primer resultado encontrado.
            if gready_search and len(requiered_money_combinations) > 0:
                return requiered_money_combinations, neighbors_order

        # Devuelve todas las combinaciones de precios requeridos.
        return requiered_money_combinations, neighbors_order

    def how_much_cost(self, resources, gready_search=True, get_neighbors_sorted=False):
        requiered_money = 0
        needed_resources = {resource: max( amount - self.resources.get(resource, 0), 0 ) for resource, amount in resources.items() }

        if sum(needed_resources.values()) == 0: return [[0]], [0]
        elif len(self.resources_selectable) > 0:
            needed_resources_for_combinations = []

            for resource_combination in self.resources_combination:
                needed_resources_for_combination = {resource: max( amount - resource_combination.get(resource, 0) - self.resources.get(resource, 0), 0 ) for resource, amount in resources.items() }

                if sum(needed_resources_for_combination.values()) == 0: return [[0]], [0]
                else:
                    # si no lo puedo construir con mis propios recursos intento comprarlos 
                    cost, neighbors_order = self.how_much_to_pay( needed_resources_for_combination, gready_search=gready_search, get_neighbors_sorted=get_neighbors_sorted)
                    if len(cost) > 0 and sum(cost[0]) <= self.resources[ResourceType.MONEY] - resources.get(ResourceType.MONEY, 0):
                        return cost, neighbors_order
        else:
            cost, neighbors_order = self.how_much_to_pay( needed_resources, gready_search=gready_search, get_neighbors_sorted=get_neighbors_sorted)
            if len(cost) > 0: # and sum(cost[0]) <= self.resources[ResourceType.MONEY] - resources.get(ResourceType.MONEY, 0):
                return cost, neighbors_order

        return [[np.inf]], [np.inf]

    def has_required_resources(self, card, use_neighbors=True):
        requiered_money = 0
        needed_resources = {resource: max( amount - self.resources.get(resource, 0), 0 ) for resource, amount in card.cost.items() }

        if sum(needed_resources.values()) == 0: 
            return True

        elif sum(needed_resources.values()) <= self.resources[ResourceType.MONEY]//2:
            cost, _ = self.how_much_cost( card.cost, gready_search=True, get_neighbors_sorted=False)
            if len(cost) > 0 and np.sum(cost[0]) <= self.resources[ResourceType.MONEY]:
                return True
        
        return False

    def print(self, v=1):
        pc = self.playable_cards()
        print( f'TOT: {self.calculate_final_score()} || ' + ' - '.join([ f'{name}:{score}' for score, name in zip(self.calculate_player_scores(), ['Mi', 'Tr', 'Wo', 'Ci', 'Co', 'Gu', 'Sc' ])]))
        print(f' Militares = {self.militar}' + f' :: Science = {self.science} - ')

        string = 'Effects : '
        for key0, item0 in self.score_effects.items():
            for key1, item1 in item0.items():
                if len(item1) > 0 and sum(item1.values()) > 0:
                    string += f' {key1}('
                    for key2, item2 in item1.items():
                        var =  {'left':'L', 'center':'C', 'right':'R'}[key2]
                        string += f'{var}' + str(item2)
                    string += ')'
        print(string)

        string = 'HAND : ' + ' - '.join([ f'{items.name}:{keys}' for items, keys in self.resources.items()])
        if v: print(string)        
        string = ' - '.join([ f'{c.name}({pc[i]})' for i, c in enumerate(self.hand) ])
        if v: print(string)        
        return string

    def choose_action(self, hand=None, Free=True, force_build=False):
        # Define la estrategia de la IA aquí, por ejemplo:
        # - Calcular el valor de cada carta en función de la situación actual del jugador
        # - Tener en cuenta la estrategia de otros jugadores y las cartas que han jugado
        # - Equilibrar la obtención de recursos, la construcción de estructuras y el progreso en la maravilla
        hand = self.hand if hand is None else hand
        action = {'Free':Free}
        if self.agent == 'random':
            action_space = ['sell']
            action_space_weight = [1] if not force_build else [0]

            if np.sum(self.playable_cards()) > 0 or Free:
                action_space += ['build']
                action_space_weight += [len(hand)]

            if self.can_build_wonder():
                action_space += ['construct']
                action_space_weight += [2] if not force_build else [0]

            action['action'] = random.choices( action_space,  weights = action_space_weight,)[0]
            if action['action'] == 'sell':       action['card_id'] = random.randint(0, len(hand)-1 )
            if action['action'] == 'build':      action['card_id'] = random.choice( np.arange(len(hand))) if Free else random.choice( np.arange(len(hand))[self.playable_cards()] ) 
            if action['action'] == 'construct':  action['card_id'] = random.randint(0, len(hand)-1 )

        return action

    def execute_action(self, action: {'action':'sell', 'card_id':None, 'Free':False}, hand=None ):
        hand = self.hand if hand is None else hand
        
        if action['action'] == 'sell': # 0 - sell a card
            self.resources[ResourceType.MONEY] += 3
            choosed_card = hand.pop( action['card_id'] )         
            self.cards_discards.append(choosed_card)

        if action['action'] == 'build': # 1 - play a card
            choosed_card = hand.pop( action['card_id'] ) 

            if self.has_chain(choosed_card) or action['Free']:
                pass

            else:
                cost, neighbors_order = self.how_much_cost(choosed_card.cost, gready_search=True, get_neighbors_sorted=True)
                MONEY_cost = np.sum(cost[0])
                self.resources[ResourceType.MONEY] -= MONEY_cost + choosed_card.cost.get(ResourceType.MONEY, 0)
                self.resources_spend[ResourceType.MONEY] += MONEY_cost + choosed_card.cost.get(ResourceType.MONEY, 0)
                for i, c in enumerate(cost[0]):
                    self.neighbors[neighbors_order[i]].resources[ResourceType.MONEY] += cost[0][i]

            self.play_card( choosed_card )

        if action['action'] == 'construct': # 2 - build a wonder
            choosed_card = hand.pop( action['card_id'] )       
            building = self.wonder.buildings[self.wonder.level]

            if not action['Free']:
                cost, neighbors_order = self.how_much_cost(building.cost, gready_search=True, get_neighbors_sorted=True)
                MONEY_cost = np.sum(cost[0])
                self.resources[ResourceType.MONEY] -= MONEY_cost + choosed_card.cost.get(ResourceType.MONEY, 0)
                self.resources_spend[ResourceType.MONEY] += MONEY_cost + choosed_card.cost.get(ResourceType.MONEY, 0)
                for i, c in enumerate(cost[0]):
                    self.neighbors[neighbors_order[i]].resources[ResourceType.MONEY] += cost[0][i]
                
            self.wonder.card_used_to_build.append(choosed_card)
            self.wonder.buildings[self.wonder.level].card_used_to_build = choosed_card
            self.wonder_effect()
            self.wonder.level_up()

        if action['action'] == 'discard': 
            choosed_card = hand.pop(action['card_id'])  
            self.cards_discards.append(choosed_card)
           
        # if has LVL2 Jardines Colgantes de Babilonia can play last card
        if len(hand) == 1:
            if self.card_effects.get('BUILD_LAST_CARD', False):
                Babilonia_action = self.choose_action()
                self.execute_action(Babilonia_action)
            else:
                last_card = hand.pop(0)  
                self.cards_discards.append(last_card)

        return hand

    def wonder_effect(self, ):
        for key, item in self.wonder.buildings[self.wonder.level].effect.items():
            if key == 'add_Militar':                        self.militar += item
            if key == 'add_MONEY':                          self.resources[ResourceType.MONEY] += item
            if key == 'add_RAW_MATERIAL':                   self.resources_selectable.append( [ResourceType.WOOD, ResourceType.STONE, ResourceType.CLAY, ResourceType.ORE] )
            if key == 'add_MANUFACTURED_GOOD':              self.resources_selectable.append( [ResourceType.LOOM, ResourceType.PAPYRUS, ResourceType.GLASS] )
            if key == 'add_SCIENCE_TEC':                    self.science['TEC'] += item
            if key == 'BUILD_LAST_CARD':                    self.card_effects['BUILD_LAST_CARD'] = True
            if key == 'FREE_BUILD':                         self.card_effects['FREE_BUILD'] = True
            if key == 'Trading_RAW_MATERIAL_left':          self.neighbors_cost[0] = item
            if key == 'Trading_RAW_MATERIAL_right':         self.neighbors_cost[1] = item
            if key == 'Trading_MANUFACTURED_GOOD_left':     self.neighbors_cost[0] = item
            if key == 'Trading_MANUFACTURED_GOOD_right':    self.neighbors_cost[1] = item
            if key == 'COPY_GUILD':                         self.card_effects['COPY_GUILD'] = True
            if key == 'FREE_DISCARD':                       self.card_effects['FREE_DISCARD'] += 1

    def card_effec(self, card):

        for key, item in card.effect.items():
            if key == 'add_Militar':                        self.militar += item
            if key == 'add_MONEY':                          self.resources[ResourceType.MONEY] += item
            if key == 'add_RAW_MATERIAL':                   self.resources_selectable.append( [ResourceType.WOOD, ResourceType.STONE, ResourceType.CLAY, ResourceType.ORE] )
            if key == 'add_MANUFACTURED_GOOD':              self.resources_selectable.append( [ResourceType.LOOM, ResourceType.PAPYRUS, ResourceType.GLASS] )
            if key == 'add_SCIENCE_TEC':                    self.science['TEC'] += item
            if key == 'BUILD_LAST_CARD':                    self.card_effects['BUILD_LAST_CARD'] = True
            if key == 'FREE_BUILD':                         self.card_effects['FREE_BUILD'] = True
            if key == 'Trading_RAW_MATERIAL_left':          self.neighbors_cost[0] = item
            if key == 'Trading_RAW_MATERIAL_right':         self.neighbors_cost[1] = item
            if key == 'Trading_MANUFACTURED_GOOD_left':     self.neighbors_cost[0] = item
            if key == 'Trading_MANUFACTURED_GOOD_right':    self.neighbors_cost[1] = item
            if key == 'COPY_GUILD':                         self.card_effects['COPY_GUILD'] = True
            if key == 'FREE_DISCARD':                       self.card_effects['FREE_DISCARD'] += 1

            if key == 'add_RAW_MATERIAL_WOOD_CLAY':         self.resources_selectable.append( [ResourceType.WOOD, ResourceType.CLAY] )
            if key == 'add_RAW_MATERIAL_STONE_CLAY':        self.resources_selectable.append( [ResourceType.STONE, ResourceType.CLAY] )
            if key == 'add_RAW_MATERIAL_CLAY_ORE':          self.resources_selectable.append( [ResourceType.CLAY, ResourceType.ORE] )
            if key == 'add_RAW_MATERIAL_STONE_WOOD':        self.resources_selectable.append( [ResourceType.WOOD, ResourceType.STONE] )
            if key == 'add_RAW_MATERIAL_WOOD_ORE':          self.resources_selectable.append( [ResourceType.WOOD, ResourceType.ORE] )
            if key == 'add_RAW_MATERIAL_ORE_STONE':         self.resources_selectable.append( [ResourceType.STONE, ResourceType.ORE] )

            if key == 'coin_RAW_MATERIAL_left':             self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.RAW_MATERIAL else 0 for c in  self.neighbors[0].cards ] )
            if key == 'coin_RAW_MATERIAL_center':           self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.RAW_MATERIAL else 0 for c in  self.cards ] )
            if key == 'coin_RAW_MATERIAL_right':            self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.RAW_MATERIAL else 0 for c in  self.neighbors[1].cards ] )
            if key == 'coin_MANUFACTURED_GOOD_left':        self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.MANUFACTURED_GOOD else 0 for c in  self.neighbors[0].cards ] )
            if key == 'coin_MANUFACTURED_GOOD_center':      self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.MANUFACTURED_GOOD else 0 for c in  self.cards ] ) 
            if key == 'coin_MANUFACTURED_GOOD_right':       self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.MANUFACTURED_GOOD else 0 for c in  self.neighbors[1].cards ] )
            if key == 'coin_COMMERCIAL_STRUCTURE_left':     self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.COMMERCIAL_STRUCTURE else 0 for c in  self.neighbors[0].cards ] )
            if key == 'coin_COMMERCIAL_STRUCTURE_center':   self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.COMMERCIAL_STRUCTURE else 0 for c in  self.cards ] ) 
            if key == 'coin_COMMERCIAL_STRUCTURE_right':    self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.COMMERCIAL_STRUCTURE else 0 for c in  self.neighbors[1].cards ] )
            if key == 'coin_SCIENTIFIC_STRUCTURE_left':     self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.SCIENTIFIC_STRUCTURE else 0 for c in  self.neighbors[0].cards ] )
            if key == 'coin_SCIENTIFIC_STRUCTURE_center':   self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.SCIENTIFIC_STRUCTURE else 0 for c in  self.cards ] ) 
            if key == 'coin_SCIENTIFIC_STRUCTURE_right':    self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.SCIENTIFIC_STRUCTURE else 0 for c in  self.neighbors[1].cards ] )
            if key == 'coin_MILITARY_STRUCTURE_left':       self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.MILITARY_STRUCTURE else 0 for c in  self.neighbors[0].cards ] )
            if key == 'coin_MILITARY_STRUCTURE_center':     self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.MILITARY_STRUCTURE else 0 for c in  self.cards ] ) 
            if key == 'coin_MILITARY_STRUCTURE_right':      self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.MILITARY_STRUCTURE else 0 for c in  self.neighbors[1].cards ] )
            if key == 'coin_GUILD_left':                    self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.GUILD else 0 for c in  self.neighbors[0].cards ] )
            if key == 'coin_GUILD_center':                  self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.GUILD else 0 for c in  self.cards ] ) 
            if key == 'coin_GUILD_right':                   self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.GUILD else 0 for c in  self.neighbors[1].cards ] )
            if key == 'coin_CIVILIAN_STRUCTURE_left':       self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.CIVILIAN_STRUCTURE else 0 for c in  self.neighbors[0].cards ] )
            if key == 'coin_CIVILIAN_STRUCTURE_center':     self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.CIVILIAN_STRUCTURE else 0 for c in  self.cards ] ) 
            if key == 'coin_CIVILIAN_STRUCTURE_right':      self.resources[ResourceType.MONEY] = item*sum( [ 1 if c.card_type == CardType.CIVILIAN_STRUCTURE else 0 for c in  self.neighbors[1].cards ] )

            if not card.card_type in self.score_effects: self.score_effects[card.card_type] = {}
            # ==== RAW_MATERIAL ==== #
            if not CardType.RAW_MATERIAL in self.score_effects[card.card_type]: self.score_effects[card.card_type][CardType.RAW_MATERIAL] = {}
            if key == 'score_RAW_MATERIAL_left':            self.score_effects[card.card_type][CardType.RAW_MATERIAL]['left']   = item + self.score_effects[card.card_type][CardType.RAW_MATERIAL].get('left',   0)
            if key == 'score_RAW_MATERIAL_center':          self.score_effects[card.card_type][CardType.RAW_MATERIAL]['center'] = item + self.score_effects[card.card_type][CardType.RAW_MATERIAL].get('center', 0)
            if key == 'score_RAW_MATERIAL_right':           self.score_effects[card.card_type][CardType.RAW_MATERIAL]['right']  = item + self.score_effects[card.card_type][CardType.RAW_MATERIAL].get('right',  0)

            # ==== MANUFACTURED_GOOD ==== #
            if not CardType.MANUFACTURED_GOOD in self.score_effects[card.card_type]: self.score_effects[card.card_type][CardType.MANUFACTURED_GOOD] = {}
            if key == 'score_MANUFACTURED_GOOD_left':       self.score_effects[card.card_type][CardType.MANUFACTURED_GOOD]['left']   = item + self.score_effects[card.card_type][CardType.MANUFACTURED_GOOD].get('left',   0)
            if key == 'score_MANUFACTURED_GOOD_center':     self.score_effects[card.card_type][CardType.MANUFACTURED_GOOD]['center'] = item + self.score_effects[card.card_type][CardType.MANUFACTURED_GOOD].get('center', 0)
            if key == 'score_MANUFACTURED_GOOD_right':      self.score_effects[card.card_type][CardType.MANUFACTURED_GOOD]['right']  = item + self.score_effects[card.card_type][CardType.MANUFACTURED_GOOD].get('right',  0)

            # ==== COMMERCIAL_STRUCTURE ==== #
            if not CardType.COMMERCIAL_STRUCTURE in self.score_effects[card.card_type]: self.score_effects[card.card_type][CardType.COMMERCIAL_STRUCTURE] = {}
            if key == 'score_COMMERCIAL_STRUCTURE_left':    self.score_effects[card.card_type][CardType.COMMERCIAL_STRUCTURE]['left']   = item + self.score_effects[card.card_type][CardType.COMMERCIAL_STRUCTURE].get('left',   0)
            if key == 'score_COMMERCIAL_STRUCTURE_center':  self.score_effects[card.card_type][CardType.COMMERCIAL_STRUCTURE]['center'] = item + self.score_effects[card.card_type][CardType.COMMERCIAL_STRUCTURE].get('center', 0)
            if key == 'score_COMMERCIAL_STRUCTURE_right':   self.score_effects[card.card_type][CardType.COMMERCIAL_STRUCTURE]['right']  = item + self.score_effects[card.card_type][CardType.COMMERCIAL_STRUCTURE].get('right',  0)

            # ==== SCIENTIFIC_STRUCTURE ==== #
            if not CardType.SCIENTIFIC_STRUCTURE in self.score_effects[card.card_type]: self.score_effects[card.card_type][CardType.SCIENTIFIC_STRUCTURE] = {}
            if key == 'score_SCIENTIFIC_STRUCTURE_left':    self.score_effects[card.card_type][CardType.SCIENTIFIC_STRUCTURE]['left']   = item + self.score_effects[card.card_type][CardType.SCIENTIFIC_STRUCTURE].get('left',   0)
            if key == 'score_SCIENTIFIC_STRUCTURE_center':  self.score_effects[card.card_type][CardType.SCIENTIFIC_STRUCTURE]['center'] = item + self.score_effects[card.card_type][CardType.SCIENTIFIC_STRUCTURE].get('center', 0)
            if key == 'score_SCIENTIFIC_STRUCTURE_right':   self.score_effects[card.card_type][CardType.SCIENTIFIC_STRUCTURE]['right']  = item + self.score_effects[card.card_type][CardType.SCIENTIFIC_STRUCTURE].get('right',  0)

            # ==== MILITARY_STRUCTURE ==== #
            if not CardType.MILITARY_STRUCTURE in self.score_effects[card.card_type]: self.score_effects[card.card_type][CardType.MILITARY_STRUCTURE] = {}
            if key == 'score_MILITARY_STRUCTURE_left':      self.score_effects[card.card_type][CardType.MILITARY_STRUCTURE]['left']   = item + self.score_effects[card.card_type][CardType.MILITARY_STRUCTURE].get('left',   0)
            if key == 'score_MILITARY_STRUCTURE_center':    self.score_effects[card.card_type][CardType.MILITARY_STRUCTURE]['center'] = item + self.score_effects[card.card_type][CardType.MILITARY_STRUCTURE].get('center', 0)
            if key == 'score_MILITARY_STRUCTURE_right':     self.score_effects[card.card_type][CardType.MILITARY_STRUCTURE]['right']  = item + self.score_effects[card.card_type][CardType.MILITARY_STRUCTURE].get('right',  0)

            # ==== GUILD ==== #
            if not CardType.GUILD in self.score_effects[card.card_type]: self.score_effects[card.card_type][CardType.GUILD] = {}
            if key == 'score_GUILD_left':                   self.score_effects[card.card_type][CardType.GUILD]['left']   = item + self.score_effects[card.card_type][CardType.GUILD].get('left',   0)
            if key == 'score_GUILD_center':                 self.score_effects[card.card_type][CardType.GUILD]['center'] = item + self.score_effects[card.card_type][CardType.GUILD].get('center', 0)
            if key == 'score_GUILD_right':                  self.score_effects[card.card_type][CardType.GUILD]['right']  = item + self.score_effects[card.card_type][CardType.GUILD].get('right',  0)

            # ==== CIVILIAN_STRUCTURE ==== #
            if not CardType.CIVILIAN_STRUCTURE in self.score_effects[card.card_type]: self.score_effects[card.card_type][CardType.CIVILIAN_STRUCTURE] = {}
            if key == 'score_CIVILIAN_STRUCTURE_left':       self.score_effects[card.card_type][CardType.CIVILIAN_STRUCTURE]['left']   = item + self.score_effects[card.card_type][CardType.CIVILIAN_STRUCTURE].get('left',   0)
            if key == 'score_CIVILIAN_STRUCTURE_center':     self.score_effects[card.card_type][CardType.CIVILIAN_STRUCTURE]['center'] = item + self.score_effects[card.card_type][CardType.CIVILIAN_STRUCTURE].get('center', 0)
            if key == 'score_CIVILIAN_STRUCTURE_right':      self.score_effects[card.card_type][CardType.CIVILIAN_STRUCTURE]['right']  = item + self.score_effects[card.card_type][CardType.CIVILIAN_STRUCTURE].get('right',  0)


            # ==== WAR_LOSS ==== #
            if not 'WAR_LOSS' in self.score_effects[card.card_type]: self.score_effects[card.card_type]['WAR_LOSS'] = {}
            if key == 'score_WAR_LOSS_left':                self.score_effects[card.card_type]['WAR_LOSS']['left']   = item + self.score_effects[card.card_type]['WAR_LOSS'].get('left',   0)
            if key == 'score_WAR_LOSS_center':              self.score_effects[card.card_type]['WAR_LOSS']['center'] = item + self.score_effects[card.card_type]['WAR_LOSS'].get('center', 0)
            if key == 'score_WAR_LOSS_right':               self.score_effects[card.card_type]['WAR_LOSS']['right']  = item + self.score_effects[card.card_type]['WAR_LOSS'].get('right',  0)

            # ==== WONDER ==== #
            if not 'WONDER' in self.score_effects[card.card_type]: self.score_effects[card.card_type]['WONDER'] = {}
            if key == 'score_WONDER_left':                  self.score_effects[card.card_type]['WONDER']['left']   = item + self.score_effects[card.card_type]['WONDER'].get('left',   0)
            if key == 'score_WONDER_center':                self.score_effects[card.card_type]['WONDER']['center'] = item + self.score_effects[card.card_type]['WONDER'].get('center', 0)
            if key == 'score_WONDER_right':                 self.score_effects[card.card_type]['WONDER']['right']  = item + self.score_effects[card.card_type]['WONDER'].get('right',  0)


        return True

    def play_card(self, card):
        self.cards.append(card)
        self.cards_names[card.name] = 0

        for key, item in card.resources_produced.items():
            self.resources[key] += item

        self.militar += card.militar
        
        for simblo, value in card.science.items():
            self.science[simblo] += value

        self.card_effec(card)

        return True

    def war(self, age):
        for neighbor in self.militares_neighbors:
            if self.militar > neighbor.militar:  self.militar_score['win']  += self.militar_points['win'][age]; self.militar_battle['win'] += 1
            if self.militar < neighbor.militar:  self.militar_score['loss'] += self.militar_points['loss'][age]; self.militar_battle['loss'] += 1
            if self.militar == neighbor.militar: self.militar_score['tie']  += self.militar_points['tie'][age]; self.militar_battle['tie'] += 1

    def calculate_military_points(self, ):
        # Calcula los puntos militares de un jugador en función de los conflictos ganados y perdidos
        return sum([ self.militar_score['win'], self.militar_score['loss'], self.militar_score['tie']  ])

    def calculate_treasury_points(self, ):
        # Calcula los puntos de tesorería (1 punto por cada 3 monedas)
        return self.resources[ResourceType.MONEY] // 3

    def calculate_wonder_points(self, ):
        # Calcula los puntos de la maravilla en función de las etapas construidas y sus bonificaciones
        return self.wonder.points # sum([ sum(items if key == 'add_POINTS' else 0 for key, items in b.effect.items() ) for b in self.wonder.buildings[:self.wonder.level] ]) # 

    def calculate_civilian_points(self, ):
        # Calcula los puntos civiles sumando los puntos de las estructuras civiles
        return sum(card.points for card in self.cards if card.card_type == CardType.CIVILIAN_STRUCTURE)

    def calculate_commercial_points(self, ):
        # Calcula los puntos comerciales en función de las estructuras comerciales y sus bonificaciones
        COMMERCIAL_STRUCTURE_score = {}
        if CardType.COMMERCIAL_STRUCTURE in self.score_effects:
            for structure_effect, structure_effect_side_dict in self.score_effects[CardType.COMMERCIAL_STRUCTURE].items():
                COMMERCIAL_STRUCTURE_score[structure_effect] = 0 
                for structure_effect_side, points in structure_effect_side_dict.items():

                    if structure_effect == 'WONDER':
                        if structure_effect_side == 'left':     COMMERCIAL_STRUCTURE_score[structure_effect] += self.neighbors[0].wonder.level * points
                        if structure_effect_side == 'right':    COMMERCIAL_STRUCTURE_score[structure_effect] += self.neighbors[1].wonder.level * points
                        if structure_effect_side == 'center':   COMMERCIAL_STRUCTURE_score[structure_effect] += self.wonder.level * points

                    if structure_effect == 'WAR_LOSS':
                        if structure_effect_side == 'left':     COMMERCIAL_STRUCTURE_score[structure_effect] += self.neighbors[0].militar_battle['loss'] * points
                        if structure_effect_side == 'right':    COMMERCIAL_STRUCTURE_score[structure_effect] += self.neighbors[1].militar_battle['right'] * points
                        if structure_effect_side == 'center':   COMMERCIAL_STRUCTURE_score[structure_effect] += self.militar_battle['loss'] * points

                    else:    
                        if structure_effect_side == 'left':     COMMERCIAL_STRUCTURE_score[structure_effect] += sum([points if card.card_type == structure_effect else 0 for card in self.neighbors[0].cards ])
                        if structure_effect_side == 'right':    COMMERCIAL_STRUCTURE_score[structure_effect] += sum([points if card.card_type == structure_effect else 0 for card in self.neighbors[1].cards ])
                        if structure_effect_side == 'center':   COMMERCIAL_STRUCTURE_score[structure_effect] += sum([points if card.card_type == structure_effect else 0 for card in self.cards ])

        else: return 0

        return sum(COMMERCIAL_STRUCTURE_score.values())

    def calculate_guild_points(self, ):
        # Calcula los puntos de gremio en función de las cartas de gremio y sus condiciones
        GUILD_score = {}
        if CardType.GUILD in self.score_effects:
            for structure_effect, structure_effect_side_dict in self.score_effects[CardType.GUILD].items():
                GUILD_score[structure_effect] = 0 
                for structure_effect_side, points in structure_effect_side_dict.items():

                    if structure_effect == 'WONDER':
                        if structure_effect_side == 'left':     GUILD_score[structure_effect] += self.neighbors[0].wonder.level * points
                        if structure_effect_side == 'right':    GUILD_score[structure_effect] += self.neighbors[1].wonder.level * points
                        if structure_effect_side == 'center':   GUILD_score[structure_effect] += self.wonder.level * points

                    if structure_effect == 'WAR_LOSS':
                        if structure_effect_side == 'left':     GUILD_score[structure_effect] += self.neighbors[0].militar_battle['loss'] * points
                        if structure_effect_side == 'right':    GUILD_score[structure_effect] += self.neighbors[1].militar_battle['loss'] * points
                        if structure_effect_side == 'center':   GUILD_score[structure_effect] += self.militar_battle['loss'] * points

                    else:    
                        if structure_effect_side == 'left':     GUILD_score[structure_effect] += sum([points if card.card_type == structure_effect else 0 for card in self.neighbors[0].cards ])
                        if structure_effect_side == 'right':    GUILD_score[structure_effect] += sum([points if card.card_type == structure_effect else 0 for card in self.neighbors[1].cards ])
                        if structure_effect_side == 'center':   GUILD_score[structure_effect] += sum([points if card.card_type == structure_effect else 0 for card in self.cards ])

        else: return 0

        return sum(GUILD_score.values())

    def calculate_science_points(self, ):
        # Calcula los puntos de ciencia en función de los sets de símbolos de ciencia y su cantidad
        vec = sorted( [ self.science['Tabla'], self.science['Engranaje'], self.science['Compass']] )[::-1] + [self.science['TEC']]
        return int(science_points_table[ tuple(vec) ])

    def calculate_player_scores(self, ) -> list:  
        # ['Mi', 'Tr', 'Wo', 'Ci', 'Co', 'Gu', 'Sc' ]
        return [self.calculate_military_points()   , self.calculate_treasury_points(), 
                self.calculate_wonder_points()     , self.calculate_civilian_points(),
                self.calculate_commercial_points() , self.calculate_guild_points(),
                self.calculate_science_points() ]

    def calculate_final_score(self) -> int:
        return np.sum( self.calculate_player_scores() )

# -------------------------------------------------------------------------- #
class Wonder:
    def __init__(self, name:str, resources:dict={}, level:int=0, buildings:list=[]):
        self.name = name
        self.resources = None

        self.level = level
        self.buildings = buildings
        self.card_used_to_build = []

        self.points = 0

    def get_side(self, ) -> str:
        return self.name[-1]

    def level_up(self, ):
        self.buildings[self.level].status = True
        self.points += sum(items if key == 'add_POINTS' else 0 for key, items in self.buildings[self.level].effect.items() )
        self.level += 1

    def level_max(self, ) -> bool:
        return self.level >= len(self.buildings)

class Building:
    def __init__(self, level:int=0, cost:dict={}, effect:dict={}):
        self.name = 'building'
        self.level = level 
        self.cost = cost
        self.effect = effect
        self.status = False
        self.card_used_to_build = None

# -------------------------------------------------------------------------- #
class SevenWondersGame:
    def __init__(self, num_players):
        self.num_players = num_players
        self.players = [Player() for _ in range(self.num_players)]
        for i, player in enumerate(self.players):  self.players[i-1].neighbors = [self.players[i-2], self.players[i]]; self.players[i-1].neighbors_cost = [2,2]
        for i, player in enumerate(self.players):  self.players[i-1].militares_neighbors = [self.players[i-2], self.players[i]]
        self.cards = self.get_basic_game_cards()
        self.wonders = self.get_basic_game_wonders()
        self.wonders_N = len(self.wonders)
        self.age = 1
        self.step = 0

        self.discard = []
        
    def reset(self, ):
        self.num_players = self.num_players
        self.players = [Player() for _ in range(self.num_players)]
        for i, player in enumerate(self.players):  self.players[i-1].neighbors = [self.players[i-2], self.players[i]]; self.players[i-1].neighbors_cost = [2,2]
        for i, player in enumerate(self.players):  self.players[i-1].militares_neighbors = [self.players[i-2], self.players[i]]
        self.cards = self.get_basic_game_cards()
        self.wonders = self.get_basic_game_wonders()
        self.wonders_N = len(self.wonders)
        self.age = 1
        self.step = 0

        self.discard = []

        return True

    def print(self, ):
        print('\n'+'*'*25+f' Step {self.step} '+f':: Age {self.age} '+'*'*25)
        print( 'Cards_discards: ' + ', '.join( [d.name for d in self.discard] )+'\n' )
        for i, p in enumerate(self.players):
            print(f'=== Player {i} === {p.wonder.name} (lvl {p.wonder.level})')
            p.print()
            for c in p.cards:
                c.print()

    def deal_wonders(self):
        wonders = [ self.wonders[w*2+random.randint(0,1)] for w in random.sample( list(np.arange(self.wonders_N//2)) , self.num_players )  ]
        for wonder, player in zip(wonders, self.players):
            player.wonder = wonder

        return True

    def deal_cards(self):
        def shuffle(card):
            random.shuffle(card)
            return [age_cards[i::len(self.players)] for i in range(len(self.players))]
        
        age_cards = [card for card in self.cards if card.age == self.age and self.num_players >= card.min_players]
        hands = shuffle(age_cards)
        for hand, player in zip(hands, self.players):
            player.hand = hand

        return True

    '''
    def play_round(self, hands):
        for _ in range(7):
            actions = []  # Esta lista almacenará las acciones que los jugadores deciden tomar
            for i, player in enumerate(self.players):
                # Aquí, la lógica de la IA o la entrada del usuario decidirán qué carta jugar o qué acción tomar
                card = self.choose_card_to_play(i, hands[i])
                action = self.choose_action(i, card)
                actions.append((i, action, card))

            for player_index, action, card in actions:
                self.execute_action(player_index, action, card)

            hands = self.pass_hands(hands)
    '''

    def play(self, random_seed=None):
        #random.seed(1 if random_seed is None else random_seed)

        self.reset()

        self.deal_wonders()
        for self.age in range(1, 4):
            hands = self.deal_cards()
            self.play_round()
            self.war()
            # Al final de cada edad, calcular puntos militares y resolver conflictos

        for player in self.players:
            if player.card_effects.get('COPY_GUILD', False) and len(self.discard) > 0:

                GUILD_list = []
                for neighbor in player.neighbors:
                    for card in neighbor.cards:
                        if card.card_type == CardType.GUILD: GUILD_list.append( card )

                if len(GUILD_list) > 0: 
                    action = player.choose_action( hand=GUILD_list, Free=True )
                    player.execute_action(action=action, hand=GUILD_list)

                player.card_effects['COPY_GUILD'] = False

        # self.calculate_final_scores()
        # Aquí puedes mostrar los puntajes finales, determinar al ganador, etc.

    def play_round(self, ):

        while len(self.players[0].hand) > 1:
            #self.print()
            self.play_step()
            self.pass_hands()
            self.step += 1
    
    def play_step(self, ):
        actions_list = []
        for player in self.players:
            actions_list.append( player.choose_action() 
                )
        
        self.discard = []
        for player, action in zip( self.players, actions_list):
            player.execute_action(action)
            self.discard += player.cards_discards

        for player in self.players:
            while player.card_effects.get('FREE_DISCARD', 0) > 0 and len(self.discard) > 0:
                action = player.choose_action( hand=self.discard, Free=True )
                player.execute_action(action=action, hand=self.discard)
                player.card_effects['FREE_DISCARD'] -= 1

    def pass_hands(self, ):

        # En las edades 1 y 3, las manos se pasan a la izquierda. En la edad 2, a la derecha.
        if self.age in [1, 3]:
            #self.players[0].hand, self.players[1].hand, *middle_hands, self.players[-1].hand = self.players[-1].hand, self.players[0].hand, *[player.hand for player in self.players[1:-1]]
            hand = [player.hand for player in self.players]
            for i, player in enumerate(self.players):
                self.players[i-1].hand = hand[i]

        elif self.age == 2:
            #self.players[-1].hand, self.players[0].hand, *middle_hands, self.players[-2].hand = self.players[0].hand, self.players[1].hand, *[player.hand for player in self.players[2:]]
            hand = [player.hand for player in self.players]
            for i, player in enumerate(self.players):
                self.players[i].hand = hand[i-1]

        return True

    def war(self, ):
        for player in self.players:
            war = player.war(self.age)

    def calculate_player_scores(self, player:object) -> list:  
        military_points = self.calculate_military_points(player)
        treasury_points = self.calculate_treasury_points(player)
        wonder_points = self.calculate_wonder_points(player)
        civilian_points = self.calculate_civilian_points(player)
        commercial_points = self.calculate_commercial_points(player)
        guild_points = self.calculate_guild_points(player)
        science_points = self.calculate_science_points(player)

        player.total_points = (military_points + treasury_points + wonder_points +
                                civilian_points + commercial_points + guild_points + science_points)

        return military_points + treasury_points + wonder_points + civilian_points + commercial_points + guild_points + science_points

    def calculate_final_scores(self):
        for player in self.players:
            military_points = self.calculate_military_points(player)
            treasury_points = self.calculate_treasury_points(player)
            wonder_points = self.calculate_wonder_points(player)
            civilian_points = self.calculate_civilian_points(player)
            commercial_points = self.calculate_commercial_points(player)
            guild_points = self.calculate_guild_points(player)
            science_points = self.calculate_science_points(player)

            player.total_points = (military_points + treasury_points + wonder_points +
                                   civilian_points + commercial_points + guild_points + science_points)

        return

    def save_game(self, name='SW_game_00', ):
        with open(str(f'{name}'), "w") as my_file:
            my_file.write(f'{self.num_players},{self.age},{self.step},{self.wonders_N}'+'\n')
            
            my_file.write(','.join([c.name for c in self.cards ])+'\n')
            my_file.write(','.join([w.name for w in self.wonders ])+'\n')
            my_file.write(','.join([d.name for d in self.discard ])+'\n')

            for player in self.players:
                my_file.write(f'{player.wonder.name}, {player.wonder.level}'+'\n')
                win, loss, tie = player.militar_battle['win'], player.militar_battle['loss'], player.militar_battle['tie']
                my_file.write(f'{win}, {loss},{tie},{player.resources[ResourceType.MONEY]}'+'\n')
                my_file.write( ','.join([h.name for h in player.hand ])+'\n' )
                my_file.write( ','.join([c.name for c in player.cards ])+'\n')

        return True

    def load_game(self, name='SW_game_00', ):
        with open(str(f'{name}'), "r") as my_file:
            pass

    def get_basic_game_wonders(self, ):
        return [
            # --- Coloso de Rodas --- #
            Wonder(name='Coloso de Rodas A', resources={ResourceType.ORE:1}, buildings=[
                Building(level=1, cost={ResourceType.WOOD:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.CLAY:3}, effect={'add_Militar':2} ),
                Building(level=3, cost={ResourceType.ORE:4} , effect={'add_POINTS':7} ) ] ), 

            Wonder(name='Coloso de Rodas B', resources={ResourceType.ORE:1}, buildings=[
                Building(level=1, cost={ResourceType.STONE:3}, effect={'add_POINTS':3, 'add_Militar':1, 'add_MONEY':3} ),
                Building(level=2, cost={ResourceType.ORE:4} , effect={'add_POINTS':4, 'add_Militar':1, 'add_MONEY':4} ) ] ), 

            # --- Faro de Alejandria --- #
            Wonder(name='Faro de Alejandria A', resources={ResourceType.GLASS:1}, buildings=[
                Building(level=1, cost={ResourceType.STONE:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.ORE:2}, effect={'add_RAW_MATERIAL':1} ),
                Building(level=3, cost={ResourceType.GLASS:2} , effect={'add_POINTS':7} ) ] ), 

            Wonder(name='Faro de Alejandria B', resources={ResourceType.GLASS:1}, buildings=[
                Building(level=1, cost={ResourceType.CLAY:2}, effect={'add_RAW_MATERIAL':3} ),
                Building(level=2, cost={ResourceType.WOOD:2}, effect={'add_MANUFACTURED_GOOD':2} ),
                Building(level=3, cost={ResourceType.STONE:3} , effect={'add_POINTS':7} ) ] ), 

            # --- El Templo de Artemisa en Efeso --- #
            Wonder(name='El Templo de Artemisa en Efeso A', resources={ResourceType.PAPYRUS:1}, buildings=[
                Building(level=1, cost={ResourceType.STONE:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.WOOD:2}, effect={'add_MONEY':9} ),
                Building(level=3, cost={ResourceType.PAPYRUS:2} , effect={'add_POINTS':7} ) ] ), 

            Wonder(name='El Templo de Artemisa en Efeso B', resources={ResourceType.PAPYRUS:1}, buildings=[
                Building(level=1, cost={ResourceType.STONE:2}, effect={'add_POINTS':2, 'add_MONEY':4} ),
                Building(level=2, cost={ResourceType.WOOD:3}, effect={'add_POINTS':3, 'add_MONEY':4} ),
                Building(level=3, cost={ResourceType.PAPYRUS:1, ResourceType.GLASS:1, ResourceType.LOOM:1} , effect={'add_POINTS':5, 'add_MONEY':4} ) ] ), 

            # --- Los Jardines Colgantes de Babilonia --- #
            Wonder(name='Los Jardines Colgantes de Babilonia A', resources={ResourceType.CLAY:1}, buildings=[
                Building(level=1, cost={ResourceType.CLAY:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.WOOD:3}, effect={'add_SCIENCE_TEC':1} ),
                Building(level=3, cost={ResourceType.CLAY:4} , effect={'add_POINTS':7} ) ] ), 

            Wonder(name='Los Jardines Colgantes de Babilonia B', resources={ResourceType.CLAY:1}, buildings=[
                Building(level=1, cost={ResourceType.CLAY:1, ResourceType.LOOM:1}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.WOOD:2, ResourceType.GLASS:1}, effect={'BUILD_LAST_CARD':1} ),
                Building(level=3, cost={ResourceType.CLAY:3, ResourceType.PAPYRUS:1} , effect={'add_SCIENCE_TEC':1} ) ] ), 

            # --- La Estatua de Zeus en Olimpia --- #
            Wonder(name='La Estatua de Zeus en Olimpia A', resources={ResourceType.WOOD:1}, buildings=[
                Building(level=1, cost={ResourceType.WOOD:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.STONE:2}, effect={'FREE_BUILD':1} ),
                Building(level=3, cost={ResourceType.ORE:2} , effect={'add_POINTS':7} ) ] ), 

            Wonder(name='La Estatua de Zeus en Olimpia B', resources={ResourceType.WOOD:1}, buildings=[
                Building(level=1, cost={ResourceType.WOOD:2}, effect={'REDUCE_TRIBUTE_LEFT':1, 'REDUCE_TRIBUTE_RIGHT':1} ),
                Building(level=2, cost={ResourceType.STONE:2}, effect={'add_POINTS':5} ),
                Building(level=3, cost={ResourceType.ORE:2, ResourceType.LOOM:1} , effect={'COPY_GUILD':1} ) ] ), 

            # --- El Mausoleo de Halicarnaso --- #
            Wonder(name='El Mausoleo de Halicarnaso A', resources={ResourceType.LOOM:1}, buildings=[
                Building(level=1, cost={ResourceType.CLAY:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.ORE:3}, effect={'FREE_DISCARD':1} ),
                Building(level=3, cost={ResourceType.LOOM:2} , effect={'add_POINTS':7} ) ] ), 

            Wonder(name='El Mausoleo de Halicarnaso B', resources={ResourceType.LOOM:1}, buildings=[
                Building(level=1, cost={ResourceType.ORE:2}, effect={'FREE_DISCARD':1, 'add_POINTS':2} ),
                Building(level=2, cost={ResourceType.CLAY:3}, effect={'FREE_DISCARD':1, 'add_POINTS':1} ),
                Building(level=3, cost={ResourceType.LOOM:1, ResourceType.PAPYRUS:1, ResourceType.GLASS:1} , effect={'FREE_DISCARD':1, } ) ] ), 

            # --- Las piramides de Giza --- #
            Wonder(name='Las piramides de Giza A', resources={ResourceType.STONE:1}, buildings=[
                Building(level=1, cost={ResourceType.STONE:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.WOOD:3}, effect={'add_POINTS':5} ),
                Building(level=3, cost={ResourceType.STONE:4} , effect={'add_POINTS':7} ) ] ), 

            Wonder(name='Las piramides de Giza B', resources={ResourceType.STONE:1}, buildings=[
                Building(level=1, cost={ResourceType.WOOD:2}, effect={'add_POINTS':3} ),
                Building(level=2, cost={ResourceType.STONE:3}, effect={'add_POINTS':5} ),
                Building(level=3, cost={ResourceType.CLAY:3}, effect={'add_POINTS':5} ),
                Building(level=4, cost={ResourceType.STONE:4, ResourceType.PAPYRUS:1} , effect={'add_POINTS':7} ) ] ), 
            ]


    def get_basic_game_cards(self, ):
        return [
            # Edad 1
            # (name, card_type, age, cost, resources_produced=None, points=0)
            Card("Lumber Yard", CardType.RAW_MATERIAL, 1, {}, {ResourceType.WOOD:1}, players=3),
            Card("Lumber Yard", CardType.RAW_MATERIAL, 1, {}, {ResourceType.WOOD:1}, players=4),
            Card("Stone Pit", CardType.RAW_MATERIAL, 1, {}, {ResourceType.STONE:1}, players=3),
            Card("Stone Pit", CardType.RAW_MATERIAL, 1, {}, {ResourceType.STONE:1}, players=5),
            Card("Clay Pool", CardType.RAW_MATERIAL, 1, {}, {ResourceType.CLAY:1}, players=3),
            Card("Clay Pool", CardType.RAW_MATERIAL, 1, {}, {ResourceType.CLAY:1}, players=5),
            Card("Ore Vein", CardType.RAW_MATERIAL, 1, {}, {ResourceType.ORE:1}, players=3),
            Card("Ore Vein", CardType.RAW_MATERIAL, 1, {}, {ResourceType.ORE:1}, players=4),
            Card("Tree Farm", CardType.RAW_MATERIAL, 1, {ResourceType.MONEY:1}, effect={'add_RAW_MATERIAL_WOOD_CLAY':1}, players=6),
            Card("Excavation", CardType.RAW_MATERIAL, 1, {ResourceType.MONEY:1}, effect={'add_RAW_MATERIAL_STONE_CLAY':1}, players=4),
            Card("Clay Pit", CardType.RAW_MATERIAL, 1, {ResourceType.MONEY:1}, effect={'add_RAW_MATERIAL_CLAY_ORE':1}, players=3),
            Card("Timber Yard", CardType.RAW_MATERIAL, 1, {ResourceType.MONEY:1}, effect={'add_RAW_MATERIAL_STONE_WOOD':1}, players=3),
            Card("Forest cave", CardType.RAW_MATERIAL, 1, {ResourceType.MONEY:1}, effect={'add_RAW_MATERIAL_WOOD_ORE':1}, players=5),
            Card("Mine", CardType.RAW_MATERIAL, 1, {ResourceType.MONEY:1}, effect={'add_RAW_MATERIAL_ORE_STONE':1}, players=6),

            Card("Glassworks", CardType.MANUFACTURED_GOOD, 1, {}, {ResourceType.GLASS:1}, players=3),
            Card("Glassworks", CardType.MANUFACTURED_GOOD, 1, {}, {ResourceType.GLASS:1}, players=6),
            Card("Press", CardType.MANUFACTURED_GOOD, 1, {}, {ResourceType.PAPYRUS:1}, players=3),
            Card("Press", CardType.MANUFACTURED_GOOD, 1, {}, {ResourceType.PAPYRUS:1}, players=6),
            Card("Loom", CardType.MANUFACTURED_GOOD, 1, {}, {ResourceType.LOOM:1}, players=3),
            Card("Loom", CardType.MANUFACTURED_GOOD, 1, {}, {ResourceType.LOOM:1}, players=6),

            Card("Pawnshop", CardType.CIVILIAN_STRUCTURE, 1, {}, points=3, players=4),
            Card("Pawnshop", CardType.CIVILIAN_STRUCTURE, 1, {}, points=3, players=7),
            Card("Baths", CardType.CIVILIAN_STRUCTURE, 1, {ResourceType.STONE:1}, points=3, players=3),
            Card("Baths", CardType.CIVILIAN_STRUCTURE, 1, {ResourceType.STONE:1}, points=3, players=7),
            Card("Altar", CardType.CIVILIAN_STRUCTURE, 1, {}, points=2, players=3),
            Card("Altar", CardType.CIVILIAN_STRUCTURE, 1, {}, points=2, players=5),
            Card("Theater", CardType.CIVILIAN_STRUCTURE, 1, {}, points=2, players=3),
            Card("Theater", CardType.CIVILIAN_STRUCTURE, 1, {}, points=2, players=6),

            Card("Tavern", CardType.COMMERCIAL_STRUCTURE, 1, {}, {ResourceType.MONEY:5}, players=3),
            Card("Tavern", CardType.COMMERCIAL_STRUCTURE, 1, {}, {ResourceType.MONEY:5}, players=5),
            Card("Tavern", CardType.COMMERCIAL_STRUCTURE, 1, {}, {ResourceType.MONEY:5}, players=7),
            Card("East Trading Post", CardType.COMMERCIAL_STRUCTURE, 1, {}, players=3, effect={'Trading_RAW_MATERIAL_left':1}),
            Card("East Trading Post", CardType.COMMERCIAL_STRUCTURE, 1, {}, players=7, effect={'Trading_RAW_MATERIAL_left':1}),
            Card("West Trading Post", CardType.COMMERCIAL_STRUCTURE, 1, {}, players=3, effect={'Trading_RAW_MATERIAL_right':1}),
            Card("West Trading Post", CardType.COMMERCIAL_STRUCTURE, 1, {}, players=7, effect={'Trading_RAW_MATERIAL_right':1}),
            Card("Marketplace", CardType.COMMERCIAL_STRUCTURE, 1, {}, players=3, effect={'Trading_MANUFACTURED_GOOD_right':1, 'Trading_MANUFACTURED_GOOD_left':1}),
            Card("Marketplace", CardType.COMMERCIAL_STRUCTURE, 1, {}, players=6, effect={'Trading_MANUFACTURED_GOOD_right':1, 'Trading_MANUFACTURED_GOOD_left':1}),

            Card("Stockade", CardType.MILITARY_STRUCTURE, 1, {ResourceType.WOOD:1}, militar=1, players=3),
            Card("Stockade", CardType.MILITARY_STRUCTURE, 1, {ResourceType.WOOD:1}, militar=1, players=7),
            Card("Barracks", CardType.MILITARY_STRUCTURE, 1, {ResourceType.ORE:1}, militar=1, players=3),
            Card("Barracks", CardType.MILITARY_STRUCTURE, 1, {ResourceType.ORE:1}, militar=1, players=5),
            Card("Guard Tower", CardType.MILITARY_STRUCTURE, 1, {ResourceType.CLAY:1}, militar=1, players=3),
            Card("Guard Tower", CardType.MILITARY_STRUCTURE, 1, {ResourceType.CLAY:1}, militar=1, players=4),

            Card("Apothecary", CardType.SCIENTIFIC_STRUCTURE, 1, {ResourceType.LOOM:1}, players=3, science={'Compass':1}),
            Card("Apothecary", CardType.SCIENTIFIC_STRUCTURE, 1, {ResourceType.LOOM:1}, players=5, science={'Compass':1}),
            Card("Workshop", CardType.SCIENTIFIC_STRUCTURE, 1, {ResourceType.GLASS:1}, players=3, science={'Engranaje':1}),
            Card("Workshop", CardType.SCIENTIFIC_STRUCTURE, 1, {ResourceType.GLASS:1}, players=7, science={'Engranaje':1}),
            Card("Scriptorium", CardType.SCIENTIFIC_STRUCTURE, 1, {ResourceType.PAPYRUS:1}, players=3, science={'Tabla':1}),
            Card("Scriptorium", CardType.SCIENTIFIC_STRUCTURE, 1, {ResourceType.PAPYRUS:1}, players=4, science={'Tabla':1}),

            # Edad 2
            Card("Sawmill", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.WOOD:2}, players=3),
            Card("Sawmill", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.WOOD:2}, players=4),
            Card("Quarry", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.STONE:2}, players=3),
            Card("Quarry", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.STONE:2}, players=4),
            Card("Brickyard", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.CLAY:2}, players=3),
            Card("Brickyard", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.CLAY:2}, players=4),
            Card("Foundry", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.ORE:2}, players=3),
            Card("Foundry", CardType.RAW_MATERIAL, 2, {ResourceType.MONEY:1}, {ResourceType.ORE:2}, players=4),

            Card("Glassworks", CardType.MANUFACTURED_GOOD, 2, {}, {ResourceType.GLASS:1}, players=3),
            Card("Glassworks", CardType.MANUFACTURED_GOOD, 2, {}, {ResourceType.GLASS:1}, players=5),
            Card("Press", CardType.MANUFACTURED_GOOD, 2, {}, {ResourceType.PAPYRUS:1}, players=3),
            Card("Press", CardType.MANUFACTURED_GOOD, 2, {}, {ResourceType.PAPYRUS:1}, players=5),
            Card("Loom", CardType.MANUFACTURED_GOOD, 2, {}, {ResourceType.LOOM:1}, players=3),
            Card("Loom", CardType.MANUFACTURED_GOOD, 2, {}, {ResourceType.LOOM:1}, players=5),

            Card("Aqueduct", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.STONE:3}, points=5, players=3, chain=['Baths']),
            Card("Aqueduct", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.STONE:3}, points=5, players=7, chain=['Baths']),
            Card("Temple", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.CLAY:1, ResourceType.GLASS:1}, points=3, players=3, chain=['Altar']),
            Card("Temple", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.CLAY:1, ResourceType.GLASS:1}, points=3, players=6, chain=['Altar']),
            Card("Statue", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.ORE:1}, points=4, players=3, chain=['Theater']),
            Card("Statue", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.ORE:1}, points=4, players=7, chain=['Theater']),
            Card("Courthouse", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.CLAY:2, ResourceType.LOOM:1}, points=4, players=3, chain=['Scriptorium']),
            Card("Courthouse", CardType.CIVILIAN_STRUCTURE, 2, {ResourceType.CLAY:2, ResourceType.LOOM:1}, points=4, players=5, chain=['Scriptorium']),

            Card("Forum", CardType.COMMERCIAL_STRUCTURE, 2, {ResourceType.CLAY:2}, players=3, chain=['East Trading Post', 'West Trading Post'], effect={'add_MANUFACTURED_GOOD':1} ),
            Card("Forum", CardType.COMMERCIAL_STRUCTURE, 2, {ResourceType.CLAY:2}, players=6, chain=['East Trading Post', 'West Trading Post'], effect={'add_MANUFACTURED_GOOD':1} ),
            Card("Forum", CardType.COMMERCIAL_STRUCTURE, 2, {ResourceType.CLAY:2}, players=7, chain=['East Trading Post', 'West Trading Post'], effect={'add_MANUFACTURED_GOOD':1} ),
            Card("Caravansery", CardType.COMMERCIAL_STRUCTURE, 2, {ResourceType.WOOD:2}, players=3, chain=['Marketplace'], effect={'add_RAW_MATERIAL':1} ),
            Card("Caravansery", CardType.COMMERCIAL_STRUCTURE, 2, {ResourceType.WOOD:2}, players=5, chain=['Marketplace'], effect={'add_RAW_MATERIAL':1} ),
            Card("Caravansery", CardType.COMMERCIAL_STRUCTURE, 2, {ResourceType.WOOD:2}, players=6, chain=['Marketplace'], effect={'add_RAW_MATERIAL':1} ),
            Card("Vineyard", CardType.COMMERCIAL_STRUCTURE, 2, {}, players=3, effect={'coin_RAW_MATERIAL_left':1, 'coin_RAW_MATERIAL_center':1, 'coin_RAW_MATERIAL_right':1,}),
            Card("Vineyard", CardType.COMMERCIAL_STRUCTURE, 2, {}, players=6, effect={'coin_RAW_MATERIAL_left':1, 'coin_RAW_MATERIAL_center':1, 'coin_RAW_MATERIAL_right':1,}),
            Card("Bazar", CardType.COMMERCIAL_STRUCTURE, 2, {}, players=4, effect={'coin_MANUFACTURED_GOOD_left':2, 'coin_MANUFACTURED_GOOD_center':2, 'coin_MANUFACTURED_GOOD_right':2,}),
            Card("Bazar", CardType.COMMERCIAL_STRUCTURE, 2, {}, players=7, effect={'coin_MANUFACTURED_GOOD_left':2, 'coin_MANUFACTURED_GOOD_center':2, 'coin_MANUFACTURED_GOOD_right':2,}),

            Card("Walls", CardType.MILITARY_STRUCTURE, 2, {ResourceType.STONE:3}, militar=2, players=3),
            Card("Walls", CardType.MILITARY_STRUCTURE, 2, {ResourceType.STONE:3}, militar=2, players=7),
            Card("Training Ground", CardType.MILITARY_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.ORE:2}, militar=2, players=4),
            Card("Training Ground", CardType.MILITARY_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.ORE:2}, militar=2, players=6),
            Card("Training Ground", CardType.MILITARY_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.ORE:2}, militar=2, players=7),
            Card("Stables", CardType.MILITARY_STRUCTURE, 2, {ResourceType.ORE:1, ResourceType.CLAY:1, ResourceType.WOOD:1}, militar=2, players=3, chain=['Apothecary']),
            Card("Stables", CardType.MILITARY_STRUCTURE, 2, {ResourceType.ORE:1, ResourceType.CLAY:1, ResourceType.WOOD:1}, militar=2, players=5, chain=['Apothecary']),
            Card("Archery Range", CardType.MILITARY_STRUCTURE, 2, {ResourceType.WOOD:2, ResourceType.ORE:1}, militar=2, players=3, chain=['Workshop']),
            Card("Archery Range", CardType.MILITARY_STRUCTURE, 2, {ResourceType.WOOD:2, ResourceType.ORE:1}, militar=2, players=6, chain=['Workshop']),

            Card("Dispensary", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.ORE:2, ResourceType.GLASS:1}, players=3, science={'Compass':1}, chain=['Apothecary']),
            Card("Dispensary", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.ORE:2, ResourceType.GLASS:1}, players=4, science={'Compass':1}, chain=['Apothecary']),
            Card("Laboratory", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.CLAY:2, ResourceType.PAPYRUS:1}, players=3, science={'Engranaje':1}, chain=['Workshop']),
            Card("Laboratory", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.CLAY:2, ResourceType.PAPYRUS:1}, players=5, science={'Engranaje':1}, chain=['Workshop']),
            Card("Library", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.STONE:2, ResourceType.LOOM:1}, players=3, science={'Tabla':1}, chain=['Scriptorium']),
            Card("Library", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.STONE:2, ResourceType.LOOM:1}, players=6, science={'Tabla':1}, chain=['Scriptorium']),
            Card("School", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.PAPYRUS:1}, players=3, science={'Tabla':1}),
            Card("School", CardType.SCIENTIFIC_STRUCTURE, 2, {ResourceType.WOOD:1, ResourceType.PAPYRUS:1}, players=7, science={'Tabla':1}),

            # Edad 3
            Card("Pantheon", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.CLAY:2, ResourceType.ORE:1, ResourceType.PAPYRUS:1, ResourceType.LOOM:1, ResourceType.GLASS:1}, points=7, players=3, chain=['Temple']),
            Card("Pantheon", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.CLAY:2, ResourceType.ORE:1, ResourceType.PAPYRUS:1, ResourceType.LOOM:1, ResourceType.GLASS:1}, points=7, players=6, chain=['Temple']),
            Card("Gardens", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.WOOD:1, ResourceType.CLAY:2}, points=5, players=3, chain=['Statue']),
            Card("Gardens", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.WOOD:1, ResourceType.CLAY:2}, points=5, players=4, chain=['Statue']),
            Card("Town Hall", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.GLASS:1, ResourceType.ORE:1, ResourceType.STONE:2}, points=6, players=3),
            Card("Town Hall", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.GLASS:1, ResourceType.ORE:1, ResourceType.STONE:2}, points=6, players=5),
            Card("Town Hall", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.GLASS:1, ResourceType.ORE:1, ResourceType.STONE:2}, points=6, players=6),
            Card("Palace", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.GLASS:1, ResourceType.PAPYRUS:1, ResourceType.LOOM:1, ResourceType.CLAY:1, ResourceType.WOOD:1, ResourceType.ORE:1, ResourceType.STONE:1}, points=8, players=3),
            Card("Palace", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.GLASS:1, ResourceType.PAPYRUS:1, ResourceType.LOOM:1, ResourceType.CLAY:1, ResourceType.WOOD:1, ResourceType.ORE:1, ResourceType.STONE:1}, points=8, players=7),
            Card("Senate", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.STONE:1, ResourceType.WOOD:2}, points=6, players=3, chain=['Library']),
            Card("Senate", CardType.CIVILIAN_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.STONE:1, ResourceType.WOOD:2}, points=6, players=5, chain=['Library']),

            Card("Haven", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.LOOM:1, ResourceType.ORE:1, ResourceType.WOOD:1}, players=3, chain=['Forum'], effect={'coin_RAW_MATERIAL_center':1, 'score_RAW_MATERIAL_center':1}),
            Card("Haven", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.LOOM:1, ResourceType.ORE:1, ResourceType.WOOD:1}, players=4, chain=['Forum'], effect={'coin_RAW_MATERIAL_center':1, 'score_RAW_MATERIAL_center':1}),
            Card("Lighthouse", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.GLASS:1, ResourceType.STONE:1}, players=3, chain=['Caravansery'], effect={'coin_COMMERCIAL_STRUCTURE_center':1, 'score_COMMERCIAL_STRUCTURE_center':1}),
            Card("Lighthouse", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.GLASS:1, ResourceType.STONE:1}, players=6, chain=['Caravansery'], effect={'coin_COMMERCIAL_STRUCTURE_center':1, 'score_COMMERCIAL_STRUCTURE_center':1}),
            Card("Chamber of Commerce", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.CLAY:2, ResourceType.PAPYRUS:1}, players=4, effect={'coin_MANUFACTURED_GOOD_center':2, 'score_MANUFACTURED_GOOD_center':2}),
            Card("Chamber of Commerce", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.CLAY:2, ResourceType.PAPYRUS:1}, players=6, effect={'coin_MANUFACTURED_GOOD_center':2, 'score_MANUFACTURED_GOOD_center':2}),
            Card("Arena", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.STONE:2}, players=3, chain=['Dispensary'], effect={'coin_WONDER_center':3, 'score_WONDER_center':1}),
            Card("Arena", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.STONE:2}, players=5, chain=['Dispensary'], effect={'coin_WONDER_center':3, 'score_WONDER_center':1}),
            Card("Arena", CardType.COMMERCIAL_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.STONE:2}, players=7, chain=['Dispensary'], effect={'coin_WONDER_center':3, 'score_WONDER_center':1}),

            Card("Fortifications", CardType.MILITARY_STRUCTURE, 3, {ResourceType.STONE:1, ResourceType.ORE:3}, militar=3, players=3, chain=['Walls']),
            Card("Fortifications", CardType.MILITARY_STRUCTURE, 3, {ResourceType.STONE:1, ResourceType.ORE:3}, militar=3, players=7, chain=['Walls']),
            Card("Circus", CardType.MILITARY_STRUCTURE, 3, {ResourceType.STONE:3, ResourceType.ORE:1}, militar=3, players=4, chain=['Training Ground']),
            Card("Circus", CardType.MILITARY_STRUCTURE, 3, {ResourceType.STONE:3, ResourceType.ORE:1}, militar=3, players=5, chain=['Training Ground']),
            Card("Circus", CardType.MILITARY_STRUCTURE, 3, {ResourceType.STONE:3, ResourceType.ORE:1}, militar=3, players=6, chain=['Training Ground']),
            Card("Arsenal", CardType.MILITARY_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.WOOD:2, ResourceType.LOOM:1}, militar=3, players=3),
            Card("Arsenal", CardType.MILITARY_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.WOOD:2, ResourceType.LOOM:1}, militar=3, players=4),
            Card("Arsenal", CardType.MILITARY_STRUCTURE, 3, {ResourceType.ORE:1, ResourceType.WOOD:2, ResourceType.LOOM:1}, militar=3, players=7),
            Card("Siege Workshop", CardType.MILITARY_STRUCTURE, 3, {ResourceType.WOOD:1, ResourceType.CLAY:3}, militar=3, players=3, chain=['Laboratory']),
            Card("Siege Workshop", CardType.MILITARY_STRUCTURE, 3, {ResourceType.WOOD:1, ResourceType.CLAY:3}, militar=3, players=5, chain=['Laboratory']),

            Card("Lodge", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.CLAY:2, ResourceType.LOOM:1, ResourceType.PAPYRUS:1}, players=3, science={'Compass':1}, chain=['Dispensary']),
            Card("Lodge", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.CLAY:2, ResourceType.LOOM:1, ResourceType.PAPYRUS:1}, players=6, science={'Compass':1}, chain=['Dispensary']),
            Card("Observatory", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.ORE:2, ResourceType.GLASS:1, ResourceType.LOOM:1}, players=3, science={'Engranaje':1}, chain=['Laboratory']),
            Card("Observatory", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.ORE:2, ResourceType.GLASS:1, ResourceType.LOOM:1}, players=7, science={'Engranaje':1}, chain=['Laboratory']),
            Card("University", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.WOOD:2, ResourceType.PAPYRUS:1, ResourceType.GLASS:1}, players=3, science={'Tabla':1}, chain=['Library']),
            Card("University", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.WOOD:2, ResourceType.PAPYRUS:1, ResourceType.GLASS:1}, players=4, science={'Tabla':1}, chain=['Library']),
            Card("Academy", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.STONE:3, ResourceType.GLASS:1}, players=3, science={'Compass':1}, chain=['School']),
            Card("Academy", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.STONE:3, ResourceType.GLASS:1}, players=7, science={'Compass':1}, chain=['School']),
            Card("Study", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.WOOD:1, ResourceType.PAPYRUS:1, ResourceType.LOOM:1}, players=3, science={'Engranaje':1}, chain=['School']),
            Card("Study", CardType.SCIENTIFIC_STRUCTURE, 3, {ResourceType.WOOD:1, ResourceType.PAPYRUS:1, ResourceType.LOOM:1}, players=5, science={'Engranaje':1}, chain=['School']),
        ] + random.sample(

            # Gremios (no se incluyen todos los gremios, ya que solo se usan algunos en cada partida)
        [ 
            Card("Workers Guild", CardType.GUILD, 3, {ResourceType.ORE:2, ResourceType.CLAY:1, ResourceType.STONE:1, ResourceType.WOOD:1}, effect={'score_RAW_MATERIAL_left':1, 'score_RAW_MATERIAL_right':1}),
            Card("Craftsmens Guild", CardType.GUILD, 3, {ResourceType.ORE:2, ResourceType.STONE:2}, effect={'score_MANUFACTURED_GOOD_left':2, 'score_MANUFACTURED_GOOD_right':2}),
            Card("Traders Guild", CardType.GUILD, 3, {ResourceType.LOOM:1, ResourceType.PAPYRUS:1, ResourceType.GLASS:1}, effect={'score_COMMERCIAL_STRUCTURE_left':1, 'score_COMMERCIAL_STRUCTURE_right':1}),
            Card("Philosophers Guild", CardType.GUILD, 3, {ResourceType.CLAY:3, ResourceType.LOOM:1, ResourceType.PAPYRUS:1}, effect={'score_SCIENTIFIC_STRUCTURE_left':1, 'score_SCIENTIFIC_STRUCTURE_right':1}),
            Card("Spies Guild", CardType.GUILD, 3, {ResourceType.CLAY:3, ResourceType.GLASS:1}, effect={'score_MILITARY_STRUCTURE_left':1, 'score_MILITARY_STRUCTURE_right':1}),
            Card("Strategists Guild", CardType.GUILD, 3, {ResourceType.ORE:2, ResourceType.STONE:1, ResourceType.LOOM:1}, effect={'score_WAR_LOSS_left':1, 'score_WAR_LOSS_right':1}),
            Card("Shipowners Guild", CardType.GUILD, 3, {ResourceType.WOOD:3, ResourceType.PAPYRUS:1, ResourceType.GLASS:1}, effect={'score_RAW_MATERIAL_center':1, 'score_MANUFACTURED_GOOD_center':1, 'score_GUILD_center':1}),
            Card("Scientists Guild", CardType.GUILD, 3, {ResourceType.WOOD:2, ResourceType.ORE:2, ResourceType.PAPYRUS:1}, science={'TEC':1}),
            Card("Magistrates Guild", CardType.GUILD, 3, {ResourceType.WOOD:3, ResourceType.STONE:1, ResourceType.LOOM:1}, effect={'score_CIVILIAN_STRUCTURE_left':1, 'score_CIVILIAN_STRUCTURE_right':1}),
            Card("Builders Guild", CardType.GUILD, 3, {ResourceType.STONE:2, ResourceType.CLAY:2, ResourceType.GLASS:1}, effect={'score_WONDER_left':1, 'score_WONDER_center':1, 'score_WONDER_right':1}),
        ],  k =self.num_players+2 )

import itertools
Nmax = 14; Vmax = 4
science_points_table = np.zeros( (Nmax,Nmax,Nmax,Vmax), dtype=np.uint32 )
for n1 in range(Nmax):
    for n2 in range( min(Nmax-n1+1, n1+1) ):
        for n3 in range( min(Nmax-n1-n2+1, n2+1)):
            for m in range(Vmax):
                for n in itertools.product([0,1,2], repeat=m):
                    vec = [n1,n2,n3]
                    for v in n: vec[v] += 1
                    points = vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2] + min(vec)*7
                    if points > science_points_table[n1][n2][n3][m]:
                        science_points_table[n1][n2][n3][m] = points


def play_game(random_seed):
    game = SevenWondersGame(4)
    game.play(random_seed=1)
    game.save_game()

if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for n in range(5):
            start_time = time.time()
            results = list(executor.map( play_game, np.arange(5) ))
            end_time = time.time()
            print(f"Time taken by the function: {end_time - start_time:.2f} seconds")


