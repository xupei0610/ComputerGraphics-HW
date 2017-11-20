#ifndef PX_CG_MAZE_HPP
#define PX_CG_MAZE_HPP

#include <vector>
#include <string>

namespace px {

struct Map
{
    std::string at;
    std::size_t width;
    std::size_t height;
    int player_x;
    int player_y;
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
    static std::vector<std::size_t> keys_id;
    static void regItems();
    Map maze;

public:
    const std::string &map;
    const std::size_t &height;
    const std::size_t &width;
    const int &player_x;
    const int &player_y;
public:
    static Map gen(std::size_t width, std::size_t height);
    static bool isWall(char e);
    static bool isEndPoint(char e);
    static bool isDoor(char e);

    Maze();
    Maze(Map const &map);
    Maze(std::size_t const &width, std::size_t const &height);

    void reset(Map const &m);
    void reset(Maze const &m);
    void reset(std::size_t const &width, std::size_t const &height);

    bool isWall(std::size_t const &index);
    bool isWall(int x, int y);
    bool isEndPoint(std::size_t const &index);
    bool isEndPoint(int x, int y);
    bool isDoor(std::size_t const &index);
    bool isDoor(int x, int y);
    bool canMoveRight();
    bool canMoveLeft();
    bool canMoveUp();
    bool canMoveDown();

    std::size_t collect(int x, int y);
    std::size_t keyFor(int x, int y);

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
