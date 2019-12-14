module fortran
implicit none
integer, allocatable ::  s(:)
integer ::  amp

contains

subroutine remove(i)
    implicit none
    integer, intent(in) ::  i  ! input
    integer ::  j,k
    integer, allocatable :: tmp(:)
    if (allocated(s)) then
        ! Check if "i" is in state s
        if ( any(s == i) ) then
            allocate(tmp(size(s)))
            tmp = s
            deallocate(s)
            allocate(s(size(tmp)-1))
            ! Find indices in s corresponding to i
            do k=1,size(tmp)
                if (tmp(k) == i) then
                    j = k
                endif
            enddo
            ! Create the new state
            s(1:j-1) = tmp(1:j-1)
            s(j:) = tmp(j+1:)
            ! Update the amplitude
            if (mod(j,2) == 0) then
                amp = -1*amp
            endif
            deallocate(tmp)
        else
            deallocate(s)
            allocate(s(0))
            amp = 0
        endif
    else
        write(*,*) "Warning: state not allocated.."
        allocate(s(0))
        amp = 0
    endif
end subroutine

end module
