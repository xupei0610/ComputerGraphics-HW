#ifndef PX_CG_GLOBAL_HPP
#define PX_CG_GLOBAL_HPP

#ifndef ASSET_PATH
#define ASSET_PATH "../asset"
#endif

#ifndef ITEM_REGISTER_CAPACITY
#define ITEM_REGISTER_CAPACITY 10000
#endif

#ifndef OBJECT_REGISTER_CAPACITY
#define OBJECT_REGISTER_CAPACITY 1000
#endif

#define ATTRIB_BIND_HELPER_WITH_TANGENT                                                             \
{                                                                                                   \
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), 0);                           \
    glEnableVertexAttribArray(0);                                                                   \
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(3*sizeof(float)));   \
    glEnableVertexAttribArray(1);                                                                   \
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(5*sizeof(float)));   \
    glEnableVertexAttribArray(2);                                                                   \
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(8*sizeof(float)));   \
    glEnableVertexAttribArray(3);                                                                   \
}

#define ATTRIB_BIND_HELPER_WITHOUT_TANGENT                                                          \
{                                                                                                   \
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), 0);                            \
    glEnableVertexAttribArray(0);                                                                   \
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void *)(3*sizeof(float)));    \
    glEnableVertexAttribArray(1);                                                                   \
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void *)(5*sizeof(float)));    \
    glEnableVertexAttribArray(2);                                                                   \
    glDisableVertexAttribArray(3);                                                                  \
}

#endif
