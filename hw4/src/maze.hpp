#ifndef PX_CG_MAZE_HPP
#define PX_CG_MAZE_HPP

#include <vector>
#include <unordered_map>
#include <string>

namespace px {
typedef char Key;
typedef char Door;

struct Map
{
    std::string at;
    int width;
    int height;
    int player_x;
    int player_y;
    std::vector<Key> collection;
    std::unordered_map<Key, std::pair<int, int> > keys;
    std::unordered_map<Door, std::pair<int, int> > doors;
};


class Maze;

namespace maze {


/**
 * gen generates a maze map
 *
 * Minimal size 100
 *
 * The returned string contains
 *
 *   : (space) empty cell
 *  @: start point
 *  $: end point
 *  a, b, c, d, e: keys
 *  A, B, C, D, E: doors
 *  {: wall ┌
 *  }: wall ┐
 *  [: wall └
 *  ]: wall ┘
 *  -: wall ─
 *  |: wall │
 *  =: wall ┬
 *  /: wall ┤
 *  \: wall ├
 *  ~: wall ┴
 *  +: wall ┼
 * @param width
 * @param height
 * @return
 */
Maze gen(std::size_t width, std::size_t height);

bool isWall(char e);
bool isEndPoint(char e);
}
}


class px::Maze
{
protected:
    static std::vector<std::pair<Key, Door> > KEYS;
    Map maze;

public:

    static const char METAL_KEY, METAL_DOOR;
    static const char WOOD_KEY, WOOD_DOOR;
    static const char WATER_KEY, WATER_DOOR;
    static const char FIRE_KEY, FIRE_DOOR;
    static const char EARTH_KEY, EARTH_DOOR;

    static const char END_POINT;
    static const char EMPTY_SLOT;
    static const char PLAYER;


    const std::string &map;
    const int &height;
    const int &width;
    const int &player_x;
    const int &player_y;
public:
    static Map gen(std::size_t width, std::size_t height);
    static bool isWall(char e);
    static bool isDoor(char e);
    static bool isKey(char e);

    Maze();
    Maze(Map const &map);
    Maze(std::size_t const &width, std::size_t const &height);

    void reset(Map const &m);
    void reset(Maze const &m);
    void reset(std::size_t const &width, std::size_t const &height);

    bool isWall(std::size_t const &index);
    bool isWall(int x, int y);
    bool isEndPoint(int x, int y);
    bool isDoor(std::size_t const &index);
    bool isDoor(int x, int y);
    bool canMoveRight();
    bool canMoveLeft();
    bool canMoveUp();
    bool canMoveDown();

    void collect(char key);
    bool canWin(int x, int y);
    void getLoc(Door d, int &x, int &y);

    void portal(int x, int y);
    void moveRight();
    void moveLeft();
    void moveUp();
    void moveDown();
    void clear(int x, int y);

    inline std::size_t index(int x, int y) const noexcept
    {
        return y*(maze.width+1)+x;
    }
    inline const char *row(int y) const
    {
        return map.data() + y*(maze.width+1);
    }
    inline char &at(int x, int y)
    {
        return maze.at.at(y*(maze.width+1)+x);
    }
    char &operator[](std::size_t const &index)
    {
        return maze.at.at(index);
    }

    ~Maze() = default;
    Maze &operator=(Maze const &m)
    {
        maze = m.maze;
        return *this;
    }
    Maze &operator=(Maze &&m)
    {
        maze = m.maze;
        return *this;
    }

};

#endif
