module ml_mod
! Machine Learning 
implicit none

integer,allocatable,dimension(:)                :: ml_input
integer,allocatable,dimension(:)                :: ml_layer_nodes
character(len=4),allocatable,dimension(:)       :: ml_layer_activation
integer,allocatable,dimension(:)                :: ml_n_layerweights
integer,allocatable,dimension(:)                :: ml_n_biasweights
real*8,allocatable,dimension(:)                 :: ml_layerweights
real*8,allocatable,dimension(:)                 :: ml_biasweights
integer                                         :: ml_last_it=0
integer                                         :: ml_n_layers
integer                                         :: ml_n_features
integer                                         :: stz
real*8, allocatable, dimension(:)               :: ml_last_kappa
real*8, allocatable, dimension(:)               :: ml_last_kappa_tmp
real*8, allocatable, dimension(:, :)            :: ml_tmp_curv
real*8, allocatable, dimension(:, :)            :: ml_last_curv

! Edge variables
integer,allocatable,dimension(:,:) :: ml_mask, ml_mask_x, ml_mask_y, ml_and_mask_x, ml_and_mask_y
real*8,allocatable,dimension(:) :: ml_sum_mask_x, ml_sum_mask_y, ml_sum_and_mask_x, ml_sum_and_mask_y
real*8,allocatable,dimension(:,:) :: ml_interp_x, ml_interp_y, ml_data_x, ml_data_y, ml_data_cut, ml_mask_cut
real*8,allocatable,dimension(:) :: ml_row, ml_and_row
! Unsharp masking variables
! real*8,allocatable,dimension(:,:) :: ml_stencil, ml_unsharp_mask, ml_difference, ml_padded_stencil
! real*8,allocatable,dimension(:, :) :: ml_interface_mask, ml_nb_mask

! real*8, allocatable, dimension(:)               :: ml_input_layer
! real*8, allocatable, dimension(:)               :: ml_tmp
! real*8, allocatable, dimension(:)               :: ml_first_layer
! real*8, allocatable, dimension(:)               :: ml_second_layer
! real*8, allocatable, dimension(:)               :: ml_third_layer
! real*8, allocatable, dimension(:)               :: ml_output_layer 
integer                                         :: loop_counter
integer                                         :: ml_bias_on, ml_edge_on, ml_mask_on

end module ml_mod
