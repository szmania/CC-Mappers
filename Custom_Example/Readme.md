# Custom Example Mapper

This is an example of a custom mapper.

## Required Files

A custom mapper requires the following file structure and files:

- `Cultures/OfficialCC_Custom_Default_Cultures.xml`: Defines the cultures for the custom mapper.
- `Factions/OfficialCC_Custom_Default_Factions.xml`: Defines the factions and units for the custom mapper.
- `Terrains/OfficialCC_Custom_Default_Terrains.xml`: Defines the terrains for the custom mapper.
- `TimePeriod.xml`: Defines the time period for the custom mapper.

## Loading Order

The loading order of mapper files (factions, cultures, terrains) is as follows:

1.  Files starting with `OfficialCC_*` are loaded first.
2.  Optional add-on files that are not submod files are loaded next, in alphabetical order.
3.  Submod files are loaded after that.
4.  Finally, optional add-ons for submods are loaded, in alphabetical order.

Files loaded last will overwrite the settings from files loaded earlier.
