import sqlite3
from src.structure import Structure
import numpy as np
from src.operations import make_sym

class Database:

    _connection: sqlite3.Connection

    def __init__(self, filename: str) -> None:
        self._connection = sqlite3.connect(filename)
        creation_stm = """
            PRAGMA foreign_keys = off;
            BEGIN TRANSACTION;
            -- Table: generation
            DROP TABLE IF EXISTS generation;
            CREATE TABLE generation (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL, population INTEGER NOT NULL DEFAULT (0), max_fitness DECIMAL NOT NULL DEFAULT (0));
            -- Table: nodes
            DROP TABLE IF EXISTS nodes;
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL, count INTEGER NOT NULL, structure_id INTEGER NOT NULL REFERENCES structures (id) ON DELETE CASCADE ON UPDATE CASCADE, x DECIMAL NOT NULL, y DECIMAL NOT NULL, Rx BOOLEAN NOT NULL DEFAULT (0), Ry BOOLEAN NOT NULL DEFAULT (0), u DECIMAL NOT NULL DEFAULT (0), v DECIMAL DEFAULT (0) NOT NULL, Px DECIMAL DEFAULT (0) NOT NULL, Py DECIMAL DEFAULT (0) NOT NULL);
            -- Table: structures
            DROP TABLE IF EXISTS structures;
            CREATE TABLE structures (id INTEGER PRIMARY KEY ON CONFLICT FAIL AUTOINCREMENT NOT NULL UNIQUE, generation_id INTEGER NOT NULL REFERENCES generation (id) ON DELETE CASCADE ON UPDATE CASCADE);
            -- Table: trusses
            DROP TABLE IF EXISTS trusses;
            CREATE TABLE trusses (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL UNIQUE, structure_id INTEGER NOT NULL REFERENCES structures (id) ON DELETE CASCADE ON UPDATE CASCADE, area DECIMAL NOT NULL, stress DECIMAL NOT NULL, fitness DECIMAL NOT NULL, length DECIMAL NOT NULL, load DECIMAL NOT NULL DEFAULT (0), start_node INTEGER REFERENCES nodes (id) ON DELETE CASCADE ON UPDATE CASCADE NOT NULL, end_node INTEGER REFERENCES nodes (id) ON DELETE CASCADE ON UPDATE CASCADE NOT NULL);
            COMMIT TRANSACTION;
            PRAGMA foreign_keys = on;"""

        self._connection.executescript(creation_stm)
        self._connection.commit()

    def commit(self) -> None:
        self._connection.commit()

    def append_generation(self, population_count: int, max_fitness: float) -> None:
        self._connection.execute(
            "INSERT INTO generation (population, max_fitness) VALUES (?,?)",
            (population_count, max_fitness),
        )
        self.commit()

    def save_structure(self, generation_id: int, structure: Structure) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            "INSERT INTO structures (generation_id) VALUES ({g})".format(
                g=generation_id
            )
        )
        structure_id = cursor.lastrowid

        # Insert nodes
        nodes = np.delete(structure._nodes, [4, 5, 9], axis=1)
        n = len(nodes)
        nodes = np.concatenate(
            [
                np.array([np.arange(len(nodes))]).T,
                nodes,
                np.array([np.full(n, structure_id, dtype=np.int32)]).T,
            ],
            axis=1,
        )
        nodes_insert = list(map(tuple, nodes.tolist()))
        cursor.executemany(
            "INSERT INTO nodes (count, x, y, u, v, Rx, Ry, Px, Py, structure_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            nodes_insert
        )

        # Insert trusses
        trusses = np.triu(structure._trusses)
        trusses_insert = []
        for s in range(0, n):
            for e in range(0, n):
                # s_id, area, load, stress, efficency, lenght, s_count, e_count
                if trusses[0, s, e] != 0:
                    trusses_insert.append(
                        (
                            trusses[1,s,e].tolist(),
                            trusses[2,s,e].tolist(),
                            trusses[3,s,e].tolist(),
                            trusses[5,s,e].tolist(),
                            trusses[6,s,e].tolist(),
                            s,
                            e,
                        )
                    )
        truss_stm = "INSERT INTO trusses (structure_id, area, load, stress, fitness, length, start_node, end_node) VALUES ({i}, ?, ?, ?, ?, ?, (SELECT id FROM nodes WHERE structure_id = {i} AND count = ?), (SELECT id FROM nodes WHERE structure_id = {i} AND count = ?))".format(i=structure_id)
        cursor.executemany(truss_stm, trusses_insert)

        self.commit()
        
        
    def read_structure(self, generation_id: int, structure_count:int) -> Structure:
        cursor = self._connection.cursor()
        generation_structures = cursor.execute("SELECT id FROM structures WHERE generation_id = {i}".format(i=generation_id)).fetchall()
        structure_id = generation_structures[structure_count][0]
        
        nodes = cursor.execute("SELECT x,y,u,v,Rx != 0 as vx,Ry != 0 as vy,Rx,Ry,Px,Py FROM nodes WHERE structure_id = {id} ORDER BY count ASC".format(id=structure_id)).fetchall()
        nodes = np.array(nodes)
        n = len(nodes)
        
        trusses = cursor.execute("SELECT (SELECT count FROM nodes WHERE id = start_node) as start, (SELECT count FROM nodes WHERE id = end_node) as end, 1 as connection, area, load, stress, 0 as t, fitness, length FROM trusses WHERE structure_id = {id} ORDER BY start ASC".format(id=structure_id)).fetchall()
        trusses_m = np.zeros((7,n,n))
        for t in range(0, len(trusses)):
            s = int(trusses[t][0])
            e = int(trusses[t][1])
            trusses_m[:,s,e] = trusses[t][2:]
            
        for l in range(0, len(trusses_m)):
            trusses_m[l] = make_sym(trusses_m[l])
        
        s = Structure()
        print(trusses_m)