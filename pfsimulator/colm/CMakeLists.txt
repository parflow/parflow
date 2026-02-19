
set (SRC_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/parflow_config.F90
  ./mkinidata/CoLMINI.F90
  ./mkinidata/MOD_Initialize.F90
  ./mkinidata/MOD_IniTimeVariable.F90
  ./mkinidata/MOD_SoilColorRefl.F90
  ./mkinidata/MOD_HtopReadin.F90
  ./mkinidata/MOD_LakeDepthReadin.F90
  ./share/MOD_Precision.F90                           
  ./share/MOD_SPMD_Task.F90                           
  ./share/MOD_Namelist.F90
  ./share/MOD_Utils.F90                               
  ./share/MOD_UserDefFun.F90                          
  ./share/MOD_TimeManager.F90 
  ./main/HYDRO/MOD_Hydro_SoilFunction.F90                                             
  ./main/MOD_Vars_Global.F90                         
  ./main/MOD_Const_Physical.F90                      
  ./main/MOD_Const_LC.F90                                                    
  ./main/MOD_Const_PFT.F90                           
  ./main/MOD_Vars_TimeInvariants.F90                 
  ./main/MOD_Vars_TimeVariables.F90                  
  ./main/MOD_Vars_1DFluxes.F90                       
  ./main/MOD_Vars_1DForcing.F90                                           
  ./main/MOD_Eroot.F90                               
  ./main/MOD_Qsadv.F90                               
  ./main/MOD_LAIEmpirical.F90                        
  ./main/MOD_NitrifData.F90                          
  ./main/MOD_NdepData.F90                            
  ./main/MOD_FireData.F90                            
  ./main/MOD_OrbCoszen.F90                           
  ./main/MOD_OrbCosazi.F90                           
  ./main/MOD_3DCanopyRadiation.F90                   
  ./main/MOD_Aerosol.F90                             
  ./main/MOD_SnowSnicar.F90                          
  ./main/MOD_Albedo.F90                              
  ./main/MOD_SnowFraction.F90                        
  ./main/MOD_MonthlyinSituCO2MaunaLoa.F90                                                     
  ./main/MOD_Irrigation.F90                          
  ./main/MOD_AssimStomataConductance.F90             
  ./main/MOD_PlantHydraulic.F90                      
  ./main/MOD_FrictionVelocity.F90                    
  ./main/MOD_TurbulenceLEddy.F90                     
  ./main/MOD_Ozone.F90                               
  ./main/MOD_CanopyLayerProfile.F90                  
  ./main/MOD_LeafTemperature.F90                     
  ./main/MOD_LeafTemperaturePC.F90                   
  ./main/MOD_SoilThermalParameters.F90               
  ./main/MOD_SoilSnowHydrology.F90                   
  ./main/MOD_SnowLayersCombineDivide.F90             
  ./main/MOD_PhaseChange.F90
  ./main/MOD_Glacier.F90                                                     
  ./main/MOD_Lake.F90                                
  ./main/MOD_SimpleOcean.F90                         
  ./main/MOD_GroundFluxes.F90                        
  ./main/MOD_GroundTemperature.F90                   
  ./main/MOD_LeafInterception.F90                    
  ./main/MOD_NetSolar.F90                            
  ./main/MOD_WetBulb.F90                             
  ./main/MOD_RainSnowTemp.F90                        
  ./main/MOD_SoilSurfaceResistance.F90               
  ./main/MOD_NewSnow.F90                             
  ./main/MOD_Thermal.F90                             
  ./main/MOD_LightningData.F90                       
  ./main/CoLMDRIVER.F90                              
  ./main/CoLMMAIN.F90                                
  ./main/CoLM.F90
  ./coupling/pf_readout_colm.F90
  ./coupling/pf_couple_colm.F90
  ./coupling/pf_getforce_colm.F90
  ./coupling/drv_gridmodule_colm.F90
  ./coupling/drv_readvegtf_colm.F90
  ./preprocess/rd_soil_properties.F90
  ./preprocess/rawdata_soil_solids_fractions.F90
  ./preprocess/rawdata_soil_thermal_parameters.F90
  )

include_directories(./include)
#set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -M${CMAKE_CURRENT_BINARY_DIR}")

add_library(pfcolm ${SRC_FILES})

install (TARGETS pfcolm DESTINATION lib)
