"""All-Peru extension of the Piura Chain-A pipeline (docs/all_peru/).

Same real-key join chain as ``crop_classifier.build_training_data`` — BD SSET
``CodigoSSET`` -> ``Grafica_Tabular/<Dept>.dta`` -> ``QGIS/<DEPT>/`` polygons — run over
every department that has all three sources, then sampled back down to Piura scale so the
modelling cost is unchanged.
"""
