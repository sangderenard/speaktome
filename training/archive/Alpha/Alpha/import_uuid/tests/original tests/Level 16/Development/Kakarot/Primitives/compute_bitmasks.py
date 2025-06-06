    def _compute_bitmasks(self, scalar_function, vector_field, isovalue):
        scalar_field = scalar_function(vector_field[...,0], vector_field[...,1], vector_field[...,2])
        """Compute bitmasks based on scalar field values."""
        effective_vertices = self.vertex_count + 1 #for centroid
        centroid_mask = (2 ** effective_vertices - 1)
        centroid_mask &= ~(1 << (effective_vertices - 1))  # Clear the highest bit (centroid bit)
        above_water = (scalar_field > isovalue).int()
        return_val = (above_water * (2 ** torch.arange(effective_vertices)))
        bitmask = return_val.sum(dim=1)
        return_val_two =  bitmask & torch.full_like(bitmask, centroid_mask)
        vector_field_relevance_mask = (bitmask > 0) & (bitmask < (2**self.vertex_count))
        return return_val_two[vector_field_relevance_mask], vector_field[vector_field_relevance_mask]