#include "maze.hpp"
#include <random>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cmath>

#include "item.hpp"

using namespace px;

std::vector<std::size_t> Maze::keys_id;

struct Wall
{
    int n1, n2;
};

struct Cell
{
    bool visited;
    bool to_r;
    bool to_b;
};

Map Maze::gen(std::size_t width, std::size_t height)
{
    auto w = static_cast<int>(width);
    auto h = static_cast<int>(height);
    auto size = w*h;

    if (size < 100)
        return {};

    static std::mt19937 sd(std::random_device{}());

    std::uniform_int_distribution<int> rd(0, size-1);

    auto maze = new Cell[size];
    std::memset(maze, 0, sizeof(Cell)*size);

    std::vector<Wall> wall;
    wall.reserve(size*4);

    auto index = rd(sd);
    maze[index].visited = true;
    maze[index].to_r = false;
    maze[index].to_b = false;

    if (index >= w)
        wall.push_back({index - w, index});
    if (index < size-w)
        wall.push_back({index, index + w});
    if (index%w > 0)
        wall.push_back({index - 1, index});
    if (index%w < w - 1)
        wall.push_back({index, index + 1});

    while (!wall.empty())
    {
        index = rd(sd) % static_cast<int>(wall.size());

        if (maze[wall[index].n1].visited ^ maze[wall[index].n2].visited)
        {
            auto tmp_index = maze[wall[index].n1].visited == false ? wall[index].n1 : wall[index].n2;
            maze[tmp_index].visited = true;

            if (wall[index].n2 - wall[index].n1 == 1)
                maze[wall[index].n1].to_r = true;
            else
                maze[wall[index].n1].to_b = true;

            if (tmp_index >= w && maze[tmp_index - w].visited == false)
                wall.push_back({tmp_index - w, tmp_index});
            if (tmp_index < size-w && maze[tmp_index+w].visited == false)
                wall.push_back({tmp_index, tmp_index + w});
            if (tmp_index%w > 0 && maze[tmp_index-1].visited == false)
                wall.push_back({tmp_index - 1, tmp_index});
            if (tmp_index%w < w - 1 && maze[tmp_index + 1].visited == false)
                wall.push_back({tmp_index, tmp_index + 1});
        }

        std::swap(wall[index], wall.back());
        wall.pop_back();
    }

    std::stringstream ss;

    ss << "{";
    for (auto j = 0; j < w-1; ++j)
        ss << (maze[j].to_r ? "--" : "-=");
    ss << "-}";

    for (auto i = 0; i < h-1; ++i)
    {
        ss << "\n|";
        for (auto j = 0; j < w; ++j)
            ss << (maze[i*width+j].to_r ? "  ": " |");
        ss << (maze[i*width].to_b ? "\n|" : "\n\\");
        for (auto j = 0; j < w-1; ++j)
        {
            if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == false &&
                maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == false)
                ss << "-+";
            else if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == false &&
                     maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == true)
                ss << "-/";
            else if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == false &&
                     maze[i*width+width+j].to_r == true && maze[i*width+j+1].to_b == false)
                ss << "-~";
            else if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == true &&
                     maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == false)
                ss << " \\";
            else if (maze[i*width+j].to_r == true       && maze[i*width+j].to_b == false &&
                     maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == false)
                ss << "-=";
            else if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == false &&
                     maze[i*width+width+j].to_r == true && maze[i*width+j+1].to_b == true)
                ss << "-]";
            else if (maze[i*width+j].to_r == true       && maze[i*width+j].to_b == false &&
                     maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == true)
                ss << "-}";
            else if (maze[i*width+j].to_r == true       && maze[i*width+j].to_b == true &&
                     maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == false)
                ss << " {";
            else if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == true &&
                     maze[i*width+width+j].to_r == true && maze[i*width+j+1].to_b == false)
                ss << " [";
            else if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == true &&
                     maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == true)
                ss << " |";
            else if (maze[i*width+j].to_r == true       && maze[i*width+j].to_b == false &&
                     maze[i*width+width+j].to_r == true && maze[i*width+j+1].to_b == false)
                ss << "--";
            else if (maze[i*width+j].to_r == true       && maze[i*width+j].to_b == true &&
                     maze[i*width+width+j].to_r == true && maze[i*width+j+1].to_b == false)
                ss << " -";
            else if (maze[i*width+j].to_r == true       && maze[i*width+j].to_b == true &&
                     maze[i*width+width+j].to_r == false && maze[i*width+j+1].to_b == true)
                ss << " |";
            else if (maze[i*width+j].to_r == true       && maze[i*width+j].to_b == false &&
                     maze[i*width+width+j].to_r == true && maze[i*width+j+1].to_b == true)
                ss << "--";
            else if (maze[i*width+j].to_r == false       && maze[i*width+j].to_b == true &&
                     maze[i*width+width+j].to_r == true && maze[i*width+j+1].to_b == true)
                ss << " |";
            else
                ss << "  ";
        }
        ss << (maze[i*w + w-1].to_b ? " |" : "-/");
    }

    ss << "\n|";
    for (auto j = 0; j < w; ++j)
        ss << (maze[size-width+j].to_r ? "  " : " |");
    ss << "\n[";
    for (auto j = 0; j < w-1; ++j)
        ss << (maze[size-width+j].to_r ? "--" : "-~");
    ss << "-]";

    Map map;
    map.height = height*2 + 1;
    map.width = width*2 + 1;
    map.at = ss.str();

    auto row = rd(sd) % (w+h) < w ? true : false;
    auto head = rd(sd) % 2 == 0 ? true : false;
    auto id = rd(sd) % (row ? h : w);
    if (row)
    {
        map.player_x = head ? 1 : w+w-1;
        map.player_y = id*2 + 1;
    }
    else
    {
        map.player_x = id*2 + 1;
        map.player_y = head ? 1 : h+h-1;
    }
//    map.player_x = 1;
//    map.player_y = 1;
//    map.at[(w+w+2)] = 'a';
//    map.at[(w+w+2)*2] = 'b';
    auto tar = map.player_y * (w+w+2) + map.player_x;
    map.at[tar] = '@';

    if (row)
    {
        if (id < h/2)
            id = h-1;
        else
            id = 0;
    }
    else if (id < w/2)
        id = w-1;
    else
        id = 0;
    head = !head;
    tar = row ? (head ? (id*2+1) * (w+w+2) : (id*2+2) * (w+w+2) - 2)
              : (head ? (id*2+1) : h*2 * (w+w+2) + (id*2+1));
    map.at[tar] = '$';

    for (auto i = 0; i < 5; ++i)
    {
        do
        {
            row = rd(sd) % (w+h) < w ? true : false;
            head = rd(sd) % 2 == 0 ? true : false;
            id = rd(sd) % (row ? h : w);
            tar = row ? (head ? (id*2+1) * (w+w+2) + 1 : (id*2+2) * (w+w+2) - 3)
                      : (head ? w+w+2 + (id*2+1) : (h*2-1) * (w+w+2) + (id*2+1));

        }
        while (map.at[tar] != ' ');
        map.at[tar] = 97 + i;

        row = !row;
        do
        {
            head = rd(sd) % 2 == 0 ? true : false;
            id = rd(sd) % (row ? h : w);
            tar = row ? (head ? (id*2+1) * (w+w+2) : (id*2+2) * (w+w+2) - 2)
                      : (head ? (id*2+1) : h*2 * (w+w+2) + (id*2+1));
        }
        while (map.at[tar] == '$' || (map.at[tar] > 64 && map.at[tar] < 'F'));
        map.at[tar] = 65 + i;
    }

    delete maze;

    return map;
}

bool Maze::isWall(char e)
{
     return e == '{' || e == '}' || e == '[' || e == ']' || e == '-' || e == '|'
         || e == '=' || e == '/' || e == '\\'|| e == '~' || e == '+';
}

bool Maze::isEndPoint(char e)
{
    return e == '$';
}

bool Maze::isDoor(char e)
{
    return e > 64 && e < 70;
}

void Maze::regItems()
{
    static bool registered = false;
    if (!registered)
    {
        keys_id.push_back(Bag::regItem("Metal Key", "", 0, false, true));
        keys_id.push_back(Bag::regItem("Wood Key",  "", 0, false, true));
        keys_id.push_back(Bag::regItem("Water Key", "", 0, false, true));
        keys_id.push_back(Bag::regItem("Fire Key",  "", 0, false, true));
        keys_id.push_back(Bag::regItem("Earth Key",  "", 0, false, true));
        registered = true;
    }
}

Maze::Maze()
        : map(maze.at), height(maze.height), width(maze.width),
          player_x(maze.player_x), player_y(maze.player_y)
{
    Maze::regItems();
}

Maze::Maze(Map const &map)
    : maze(map),
      map(maze.at), height(maze.height), width(maze.width),
      player_x(maze.player_x), player_y(maze.player_y)
{
    Maze::regItems();
}

Maze::Maze(std::size_t const &width, std::size_t const &height)
    : maze(Maze::gen(width, height)),
      map(maze.at), height(maze.height), width(maze.width),
      player_x(maze.player_x), player_y(maze.player_y)
{
    Maze::regItems();
}

void Maze::reset(Map const &m)
{
    maze = m;
}
void Maze::reset(Maze const &m)
{
    maze = m.maze;
}

void Maze::reset(std::size_t const &width, std::size_t const &height)
{
    maze = Maze::gen(width, height);
}

bool Maze::isWall(int x, int y)
{
    if (x < 0 || x >= static_cast<int>(width) || y < 0 || y >= static_cast<int>(height))
        return true;
    return Maze::isWall(at(x, y));
}

bool Maze::isEndPoint(int x, int y)
{
    if (x < 0 || x >= static_cast<int>(width) || y < 0 || y >= static_cast<int>(height))
        return false;
    return Maze::isEndPoint(at(x, y));
}

bool Maze::isDoor(int x, int y)
{
    if (x < 0 || x >= static_cast<int>(width) || y < 0 || y >= static_cast<int>(height))
        return false;
    return Maze::isDoor(at(x, y));
}

bool Maze::isWall(std::size_t const &index)
{
    return Maze::isWall(operator[](index));
}

bool Maze::isEndPoint(std::size_t const &index)
{
    return Maze::isEndPoint(operator[](index));
}

bool Maze::isDoor(std::size_t const &index)
{
    return Maze::isDoor(operator[](index));
}

bool Maze::canMoveRight()
{
    return !Maze::isWall(player_x+1, player_y);
}

bool Maze::canMoveLeft()
{
    return !Maze::isWall(player_x-1, player_y);
}

bool Maze::canMoveUp()
{
    return !Maze::isWall(player_x, player_y-1);
}

bool Maze::canMoveDown()
{
    return !Maze::isWall(player_x, player_y+1);
}

std::size_t Maze::collect(int x, int y)
{
    auto e = at(x, y);
    if (e > 96 && e < 102)
        return keys_id[e-97];
    return 0;
}

std::size_t Maze::keyFor(int x, int y)
{
    auto e = at(x, y);
    if (e > 64 && e < 70)
        return keys_id[e-65];
    return 0;
}

void Maze::portal(int x, int y)
{
    if (at(player_x, player_y) == '@')
        at(player_x, player_y) = ' ';
    maze.player_x = x;
    maze.player_y = y;
    auto & e = at(player_x, player_y);
    if (Maze::isWall(e) || e == ' ')
        at(x, y) = '@';
}

void Maze::moveRight()
{
    portal(maze.player_x+1, maze.player_y);
}

void Maze::moveLeft()
{
    portal(maze.player_x-1, maze.player_y);
}

void Maze::moveUp()
{
    portal(maze.player_x, maze.player_y-1);
}

void Maze::moveDown()
{
    portal(maze.player_x, maze.player_y+1);
}

void Maze::clear(int x, int y)
{
    auto & e = at(x, y);
    if (x == player_x && y == player_y)
        e = '@';
    else if (e != ' ')
        e = ' ';
}