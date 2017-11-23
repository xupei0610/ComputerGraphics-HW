#include "maze.hpp"
#include <random>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cmath>

#include "item.hpp"

using namespace px;

const char Maze::METAL_KEY = 'a'; const char Maze::METAL_DOOR = 'A';
const char Maze::WOOD_KEY  = 'b'; const char Maze::WOOD_DOOR  = 'B';
const char Maze::WATER_KEY = 'c'; const char Maze::WATER_DOOR = 'C';
const char Maze::FIRE_KEY  = 'd'; const char Maze::FIRE_DOOR  = 'D';
const char Maze::EARTH_KEY = 'e'; const char Maze::EARTH_DOOR = 'E';

const char Maze::END_POINT = '$';
const char Maze::PLAYER = '@';
const char Maze::EMPTY_SLOT = ' ';

std::vector<std::pair<Key, Door> > Maze::KEYS = {
        {METAL_KEY, METAL_DOOR},
        {WOOD_KEY,  WOOD_DOOR},
        {WATER_KEY, WATER_DOOR},
        {FIRE_KEY,  FIRE_DOOR},
        {EARTH_KEY, EARTH_DOOR},
};

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
    map.player_x = 1;
    map.player_y = 1;
    auto tar = map.player_y * (w+w+2) + map.player_x;
    map.at[tar] = PLAYER;

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
    map.at[tar] = END_POINT;

    for (auto & k : Maze::KEYS)
//    for (auto i = 0; i < 5; ++i)
    {
        do
        {
            row = rd(sd) % (w+h) < w ? true : false;
            head = rd(sd) % 2 == 0 ? true : false;
            id = rd(sd) % (row ? h : w);
            tar = row ? (head ? (id*2+1) * (w+w+2) + 1 : (id*2+2) * (w+w+2) - 3)
                      : (head ? w+w+2 + (id*2+1) : (h*2-1) * (w+w+2) + (id*2+1));

        }
        while (map.at[tar] != EMPTY_SLOT);
        map.at[tar] = k.first;
        map.keys[k.first] = std::make_pair(tar % (w+w+2), tar/(w+w+2));

        row = !row;
        do
        {
            head = rd(sd) % 2 == 0 ? true : false;
            id = rd(sd) % (row ? h : w);
            tar = row ? (head ? (id*2+1) * (w+w+2) : (id*2+2) * (w+w+2) - 2)
                      : (head ? (id*2+1) : h*2 * (w+w+2) + (id*2+1));
        }
        while (map.at[tar] == END_POINT || !Maze::isWall(map.at[tar]));
        map.at[tar] = k.second;
        map.doors[k.second] = std::make_pair(tar % (w+w+2), tar/(w+w+2));
    }

    delete maze;

    return map;
}

bool Maze::isWall(char e)
{
     return e == '{' || e == '}' || e == '[' || e == ']' || e == '-' || e == '|'
         || e == '=' || e == '/' || e == '\\'|| e == '~' || e == '+';
}

bool Maze::isDoor(char e)
{
    for (const auto &k : KEYS)
    {
        if (e == k.second)
            return true;
    }
    return false;
}

bool Maze::isKey(char e)
{
    for (const auto &k : KEYS)
    {
        if (e == k.first)
            return true;
    }
    return false;
}

Maze::Maze()
        : map(maze.at), height(maze.height), width(maze.width),
          player_x(maze.player_x), player_y(maze.player_y)
{}

Maze::Maze(Map const &map)
    : maze(map),
      map(maze.at), height(maze.height), width(maze.width),
      player_x(maze.player_x), player_y(maze.player_y)
{}

Maze::Maze(std::size_t const &width, std::size_t const &height)
    : maze(Maze::gen(width, height)),
      map(maze.at), height(maze.height), width(maze.width),
      player_x(maze.player_x), player_y(maze.player_y)
{}

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
//
//    maze.at[1] = METAL_KEY;
//    maze.at[2] = WOOD_KEY;
//    maze.at[3] = WATER_KEY;
//    maze.at[4] = FIRE_KEY;
//    maze.at[5] = EARTH_KEY;
//    collect(METAL_KEY);
//    collect(WOOD_KEY);
//    collect(WATER_KEY);
//    collect(FIRE_KEY);
//    collect(EARTH_KEY);

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
    return at(x, y) == END_POINT;
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

bool Maze::canWin(int x, int y)
{
    if (isEndPoint(x, y))
        return true;

    if (maze.collection.empty())
        return false;

    auto e = at(x, y);
    for (const auto &p : KEYS)
    {
        if (e == p.second)
        {
            for (const auto &i : maze.collection)
            {
                if (i == p.first)
                    return true;
            }
        }
    }
    return false;
}

void Maze::collect(char key)
{
    if (Maze::isKey(key))
    {
        maze.collection.push_back(key);
        clear(maze.keys[key].first, maze.keys[key].second);
    }
}

void Maze::getLoc(Door d, int &x, int &y)
{
    auto it = maze.doors.find(d);
    if (it == maze.doors.end())
    {
        x = -1; y = -1;
    }
    else
    {
        x = it->second.first;
        y = it->second.second;
    }
}

void Maze::portal(int x, int y)
{
    if (at(player_x, player_y) == '@')
        at(player_x, player_y) = ' ';
    maze.player_x = x < 0 ? 0 : x < maze.width ? x : maze.width - 1;
    maze.player_y = y < 0 ? 0 : y < maze.height ? y : maze.height - 1;
    auto & e = at(player_x, player_y);
    if (e == ' ')
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