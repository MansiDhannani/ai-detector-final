37. #A* Pathfinding Algorithm
import heapq

def a_star(start, goal, graph, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))

    g_cost = {start: 0}
    parent = {start: None}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for neighbor, cost in graph[current]:
            new_cost = g_cost[current] + cost
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic(neighbor)
                heapq.heappush(open_set, (f_cost, neighbor))
                parent[neighbor] = current

    return None