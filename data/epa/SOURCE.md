# EPA state land-use greenhouse-gas inventory

- Publisher: US Environmental Protection Agency, *Inventory of U.S. Greenhouse Gas Emissions and Sinks by State: 1990-2022*
  (EPA-430-R-24-006). US government work, public domain.
- Page: https://www.epa.gov/ghgemissions/state-ghg-emissions-and-removals
- File: https://www.epa.gov/system/files/other-files/2024-09/allstateghgdata90-22_v082924.zip (downloaded 2026-10-21)
- sha256 of the zip: `2f12c30eadd5ea663ccce47802c382131a3777410998c44d320fcb5fcfb39cf4`
- Workbook inside: `AllStateGHGData90-22_v082924.xlsx`, tab *Data by UNFCCC-IPCC Sectors*.

`lulucf_by_state.csv` is made from it by `python -m etl.epa_inventory <path to the xlsx>`: the sector *Land Use, Land-Use Change and
Forestry*, summed over categories, gases (CO2, CH4, N2O as CO2 equivalent, AR5 100-year GWPs) and carbon pools, for AL, CA, TX, WA and
FL, 1990-2022. Units: Tg (million metric tons) CO2e; emissions positive, removals negative.

EPA land category to the pipeline's land type: Forest Land = forest, Cropland = agriculture, Grassland = shrubland,
Settlements = urban, Wetlands = wetland. EPA notes that state estimates exclude harvested wood products and that summed state totals
differ from the national total.
